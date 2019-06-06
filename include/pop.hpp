
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
#include <limits>
#include <random>
#include <sax/iostream.hpp>
#include <span>
#include <vector>

#include <sax/uniform_int_distribution.hpp>

#include "fcc.hpp"
#include "rng.hpp"
#include "uniformly_decreasing_discrete_distribution.hpp"

#include <plf/plf_nanotimer.h>

template<int PopSize, int NumInput, int NumNeurons, int NumOutput>
struct Population {

    static constexpr int BreedSize = PopSize / 2;

    using Network = FullyConnectedNeuralNetwork<NumInput, NumNeurons, NumOutput>;

    struct Individual {

        float fitness;
        int age = 0;
        Network * id;

        [[nodiscard]] bool operator== ( Individual const & rhs_ ) const noexcept { return rhs_.id == id; }
        [[nodiscard]] bool operator!= ( Individual const & rhs_ ) const noexcept { return not operator== ( rhs_ ); }

        template<typename Stream>
        [[maybe_unused]] friend Stream & operator<< ( Stream & out_, Individual const & f_ ) noexcept {
            out_ << '<' << f_.id << ' ' << f_.age << ' ' << f_.fitness << '>';
            return out_;
        }
    };

    std::vector<Individual> m_population{ PopSize };

    Population ( ) {
        std::for_each ( std::execution::par_unseq, std::begin ( m_population ), std::end ( m_population ),
                        []( Individual & i ) { i.id = new Network ( ); } );
    }

    ~Population ( ) noexcept {
        std::for_each ( std::execution::par_unseq, std::begin ( m_population ), std::end ( m_population ),
                        []( Individual & i ) noexcept { delete i.id; } );
    }

    void evaluate ( ) noexcept {
        std::for_each ( std::execution::par_unseq, std::begin ( m_population ), std::end ( m_population ),
                        []( Individual & i ) noexcept {
                            i.fitness = i.id->run ( );
                            i.age += 1;
                        } );
        std::sort ( std::execution::par_unseq, std::begin ( m_population ), std::end ( m_population ),
                    []( Individual const & a, Individual const & b ) noexcept { return a.fitness > b.fitness; } );
    }

    void mutate ( Network * c_ ) noexcept {
        int const mup  = std::uniform_int_distribution<int> ( 0, Network::NumWeights - 1 ) ( Rng::gen ( ) ); // mutation point.
        ( *c_ )[ mup ] = std::normal_distribution<float> ( 0.0f, 1.0f ) ( Rng::gen ( ) );
    }

    void crossover ( std::tuple<Network const &, Network const &> p_, Network * c_ ) noexcept {
        int const cop = std::uniform_int_distribution<int> ( 0, Network::NumWeights - 2 ) ( Rng::gen ( ) ); // crossover point.
        std::copy ( std::begin ( std::get<0> ( p_ ) ), std::begin ( std::get<0> ( p_ ) ) + cop, std::begin ( *c_ ) );
        std::copy ( std::begin ( std::get<1> ( p_ ) ) + cop, std::end ( std::get<1> ( p_ ) ), std::begin ( *c_ ) + cop );
    }

    void reproduce ( ) noexcept {
        plf::nanotimer t;
        t.start ( );
        std::for_each ( std::execution::par_unseq, std::begin ( m_population ) + BreedSize,
                        std::end ( m_population ), [this]( Individual & i ) noexcept {
                            crossover ( random_couple ( ), i.id );
                            if ( Rng::bernoulli ( 0.05 ) )
                                mutate ( i.id );
                        } );
        std::cout << t.get_elapsed_us ( ) / PopSize << " microseconds per network" << nl;
    }

    [[nodiscard]] static int sample ( ) noexcept { return uniformly_decreasing_discrete_distribution<BreedSize>{}( Rng::gen ( ) ); }
    [[nodiscard]] static std::tuple<int, int> sample_match ( ) noexcept {
        auto g                 = []( ) { return uniformly_decreasing_discrete_distribution<BreedSize>{}( Rng::gen ( ) ); };
        std::tuple<int, int> r = { g ( ), g ( ) };
        while ( std::get<0> ( r ) == std::get<1> ( r ) )
            std::get<1> ( r ) = g ( );
        return r;
    }

    [[nodiscard]] Network const & random_parent ( ) const noexcept { return *m_population[ sample ( ) ].id; }
    [[nodiscard]] std::tuple<Network const &, Network const &> random_couple ( ) const noexcept {
        auto [ p0, p1 ] = sample_match ( );
        return { *m_population[ p0 ].id, *m_population[ p1 ].id };
    }
};
