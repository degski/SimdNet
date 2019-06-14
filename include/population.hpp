
// MIT License
//
// Copyright (c) 2019 degski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <execution>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sax/iostream.hpp>
#include <span>
#include <vector>

#include <sax/uniform_int_distribution.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>

#include "fcc.hpp"
#include "globals.hpp"
#include "rng.hpp"
#include "snake.hpp"
#include "uniformly_decreasing_discrete_distribution_vose.hpp"

#include <plf/plf_nanotimer.h>

template<int PopSize, int FieldSize, int NumInput, int NumNeurons, int NumOutput>
struct Population {

    static constexpr int BreedSize = PopSize / 8;

    using TheBrain   = FullyConnectedNeuralNetwork<NumInput, NumNeurons, NumOutput>;
    using SnakeSpace = SnakeSpace<FieldSize, NumInput, NumNeurons, NumOutput>;

    struct Individual {

        float fitness;
        int age = 0;
        TheBrain * id;

        [[nodiscard]] bool operator== ( Individual const & rhs_ ) const noexcept { return rhs_.id == id; }
        [[nodiscard]] bool operator!= ( Individual const & rhs_ ) const noexcept { return not operator== ( rhs_ ); }

        template<typename Stream>
        [[maybe_unused]] friend Stream & operator<< ( Stream & out_, Individual const & f_ ) noexcept {
            out_ << '<' << f_.id << ' ' << f_.age << ' ' << f_.fitness << '>';
            return out_;
        }

        private:
        friend class cereal::access;

        template<class Archive>
        void save ( Archive & ar_ ) const {
            ar_ ( fitness );
            ar_ ( age );
            ar_ ( *id );
        }

        template<class Archive>
        void load ( Archive & ar_ ) {
            ar_ ( fitness );
            ar_ ( age );
            id = new TheBrain ( );
            ar_ ( *id );
        }
    };

    Population ( ) {
        std::for_each ( std::execution::par_unseq, std::begin ( m_population ), std::end ( m_population ),
                        [] ( Individual & i ) { i.id = new TheBrain ( ); } );
    }

    ~Population ( ) noexcept {
        std::for_each ( std::execution::par_unseq, std::begin ( m_population ), std::end ( m_population ),
                        [] ( Individual & i ) noexcept { delete i.id; } );
    }

    void evaluate ( ) noexcept {
        static thread_local SnakeSpace snake_space;
        std::for_each ( std::execution::par_unseq, std::begin ( m_population ), std::end ( m_population ),
            [] ( Individual & i ) noexcept {
                i.fitness += ( ( snake_space.run ( i.id ) - i.fitness ) / static_cast<float> ( ++i.age ) ); // Maintain the average.
            } );
        std::sort ( std::execution::par_unseq, std::begin ( m_population ), std::end ( m_population ),
                    [] ( Individual const & a, Individual const & b ) noexcept { return a.fitness > b.fitness; } );
        // save_to_file_bin ( *this, "z://tmp", "population" );
    }

    void mutate ( TheBrain * const c_ ) noexcept {
        int const mup = std::uniform_int_distribution<int> ( 0, TheBrain::NumWeights - 1 ) ( Rng::gen ( ) ); // mutation point.
        ( *c_ )[ mup ] += std::normal_distribution<float> ( 0.0f, 1.0f ) ( Rng::gen ( ) );
    }

    void crossover ( std::tuple<TheBrain const &, TheBrain const &> p_, TheBrain * const c_ ) noexcept {
        int const cop = std::uniform_int_distribution<int> ( 0, TheBrain::NumWeights - 2 ) ( Rng::gen ( ) ); // crossover point.
        std::copy ( std::begin ( std::get<0> ( p_ ) ), std::begin ( std::get<0> ( p_ ) ) + cop, std::begin ( *c_ ) );
        std::copy ( std::begin ( std::get<1> ( p_ ) ) + cop, std::end ( std::get<1> ( p_ ) ), std::begin ( *c_ ) + cop );
    }

    void reproduce ( ) noexcept {
        std::for_each ( std::execution::par_unseq, std::begin ( m_population ) + BreedSize, std::end ( m_population ),
                        [this] ( Individual & i ) noexcept {
                            crossover ( random_couple ( ), i.id );
                            if ( Rng::bernoulli ( 0.05 ) )
                                mutate ( i.id );
                            i.fitness = 0.0f;
                            i.age = 0;
                        } );
    }

    [[nodiscard]] static int sample ( ) noexcept { return uniformly_decreasing_discrete_distribution<BreedSize>{}( Rng::gen ( ) ); }
    [[nodiscard]] static std::tuple<int, int> sample_match ( ) noexcept {
        auto g = [] ( ) noexcept { return uniformly_decreasing_discrete_distribution<BreedSize>{}( Rng::gen ( ) ); };
        std::tuple<int, int> r = { g ( ), g ( ) };
        while ( std::get<0> ( r ) == std::get<1> ( r ) )
            std::get<1> ( r ) = g ( );
        return r;
    }

    [[nodiscard]] TheBrain const & random_parent ( ) const noexcept { return *m_population[ sample ( ) ].id; }
    [[nodiscard]] std::tuple<TheBrain const &, TheBrain const &> random_couple ( ) const noexcept {
        auto [ p0, p1 ] = sample_match ( );
        return { *m_population[ p0 ].id, *m_population[ p1 ].id };
    }

    [[nodiscard]] float average_fitness ( ) const noexcept {
        return std::transform_reduce ( std::execution::par_unseq, std::begin ( m_population ),
                                       std::begin ( m_population ) + BreedSize, 0.0f, std::plus<> ( ),
                                       [] ( Individual const & i ) noexcept { return i.fitness; } ) /
               static_cast<float> ( BreedSize );
    }

    [[nodiscard]] float average_age ( ) const noexcept {
        return static_cast<float> ( std::transform_reduce ( std::execution::par_unseq, std::begin ( m_population ),
                                       std::begin ( m_population ) + BreedSize, 0, std::plus<> ( ),
                                       []( Individual const & i ) noexcept { return i.age; } ) ) /
               static_cast<float> ( BreedSize );
    }

    void run ( ) noexcept {
        while ( true ) {
            evaluate ( );
            ++m_generation;
            if ( m_generation > 0 ) {
                SnakeSpace snake_space;
                snake_space.run_display ( m_population[ 0 ].id );
                float const af = average_fitness ( );
                float const aa = average_age ( );
                std::wcout << L" generation " << std::setw ( 6 ) << m_generation << L" fitness " << std::setprecision ( 2 )
                           << std::fixed << std::setw ( 7 ) << m_population[ 0 ].fitness << L" " << m_population[ 0 ].age << " ("
                           << std::setw ( 7 ) << af << L" " << aa << ")" << nl;
            }
            reproduce ( );
        }
    }

    private:
    friend class cereal::access;

    template<class Archive>
    void serialize ( Archive & ar_ ) {
        ar_ ( m_population );
        ar_ ( m_generation );
    }

    std::vector<Individual> m_population{ PopSize };
    int m_generation = 0;
};
