
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

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>

#include "fcc.hpp"
#include "globals.hpp"
#include "rng.hpp"
#include "snake.hpp"
#include "uniformly_decreasing_discrete_distribution_vose.hpp"

#include <plf_nanotimer.h>

struct ConfigParams {
    bool display_match;
    bool save_population;
    bool load_population;

    private:
    friend class cereal::access;

    template<class Archive>
    void serialize ( Archive & ar_ ) {
        ar_ ( CEREAL_NVP ( display_match ) );
        ar_ ( CEREAL_NVP ( save_population ) );
        ar_ ( CEREAL_NVP ( load_population ) );
    }
};

struct Config final {
    Config ( Config && )      = delete;
    Config ( Config const & ) = delete;

    Config & operator= ( Config && ) = delete;
    Config & operator= ( Config const & ) = delete;

    [[nodiscard]] static ConfigParams & instance ( ) noexcept {
        static ConfigParams params;
        return params;
    }

    [[maybe_unused]] static ConfigParams & load ( ) noexcept {
        load_from_file_json ( s_name, instance ( ), s_file );
        return instance ( );
    }
    [[maybe_unused]] static ConfigParams & save ( ) noexcept {
        save_to_file_json ( s_name, instance ( ), s_file );
        return instance ( );
    }

    static constexpr char const s_file[]{ "z://tmp//config.json" };
    static constexpr char const s_name[]{ "config" };
};

template<int PopSize, int FieldSize, int NumInput, int NumNeurons, int NumOutput>
struct Population {

    static constexpr int BreedSize = PopSize / 3;

    using TheBrain   = FullyConnectedNeuralNetwork<NumInput, NumNeurons, NumOutput>;
    using SnakeSpace = SnakeSpace<FieldSize, NumInput, NumNeurons, NumOutput>;

    // This is a 'dumb' object, no memory is managed, but memory is
    // created on a load iff required.
    struct Individual {

        float fitness;
        int age       = 0;
        TheBrain * id = nullptr;

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
            if ( nullptr == id )
                id = new TheBrain ( );
            ar_ ( *id );
        }
    };

    Population ( ) {
        if ( Config::load ( ).load_population ) {
            load ( );
        }
        else { // load_population == false, load next time.
            Config::instance ( ).load_population = true;
            Config::save ( );
            std::for_each ( std::execution::par_unseq, std::begin ( m_population ), std::end ( m_population ),
                            [] ( Individual & i ) {
                                if ( nullptr == i.id )
                                    i.id = new TheBrain ( );
                            } );
        }
    }

    ~Population ( ) noexcept {
        std::for_each ( std::execution::par_unseq, std::begin ( m_population ), std::end ( m_population ),
                        [] ( Individual & i ) noexcept {
                            delete i.id;
                            i.id = nullptr;
                        } );
    }

    void evaluate ( ) noexcept {
        static thread_local SnakeSpace snake_space;
        std::for_each (
            std::execution::par_unseq, std::begin ( m_population ), std::end ( m_population ), [] ( Individual & i ) noexcept {
                ++i.age;
                i.fitness +=
                    ( ( snake_space.run ( i.id, i.age ) - i.fitness ) / static_cast<float> ( i.age ) ); // Maintain the average.
            } );

        std::sort ( std::execution::par_unseq, std::begin ( m_population ), std::end ( m_population ),
                    [] ( Individual const & a, Individual const & b ) noexcept { return a.fitness > b.fitness; } );

        // print_fitness ( );
        // std::wcout << nl << nl;
    }

    void mutate ( TheBrain * const c_ ) noexcept {
        static uniformly_decreasing_discrete_distribution<4> dddis;
        // static std::piecewise_linear_distribution<float> tridis = triangular_distribution ( );
        int rep = dddis ( Rng::gen ( ) );
        do {
            int const mup = std::uniform_int_distribution<int> ( 0, TheBrain::NumWeights - 1 ) ( Rng::gen ( ) ); // mutation point.
            ( *c_ )[ mup ] += std::normal_distribution<float> ( 0.0f, 2.0f ) ( Rng::gen ( ) );
            // ( *c_ )[ mup ] += tridis ( Rng::gen ( ) );
        } while ( rep-- );
    }

    void reproduce ( ) noexcept {
        std::for_each ( std::execution::par_unseq, std::begin ( m_population ) + BreedSize, std::end ( m_population ),
                        [this] ( Individual & i ) noexcept {
                            TheBrain const & parent = random_parent ( );
                            std::copy ( std::begin ( parent ), std::end ( parent ), std::begin ( *i.id ) );
                            mutate ( i.id );
                            i.fitness = 0.0f;
                            i.age     = 0;
                        } );
        // print_fitness ( );
        // std::wcout << nl << nl;
    }

    [[nodiscard]] TheBrain const & random_parent ( ) const noexcept { return *m_population[ sample ( ) ].id; }
    [[nodiscard]] std::tuple<TheBrain const &, TheBrain const &> random_couple ( ) const noexcept {
        auto [ p0, p1 ] = sample_match ( );
        return { *m_population[ p0 ].id, *m_population[ p1 ].id };
    }

    void display ( ) const noexcept {
        cls ( );
        SnakeSpace snake_space;
        snake_space.run_display ( m_population[ 0 ].id );
    }

    void print_statistics ( ) const noexcept {
        float const af = average_fitness ( );
        float const aa = average_age ( );
        std::wcout << L" generation " << std::setw ( 6 ) << m_generation << L" fitness " << std::setprecision ( 2 ) << std::fixed
                   << std::setw ( 7 ) << m_population[ 0 ].fitness << L" " << m_population[ 0 ].age << L" (" << std::setw ( 7 )
                   << af << L" " << aa << L")" << nl;
    }

    void run ( ) noexcept {
        static ConfigParams const & config = Config::instance ( );
        while ( true ) {
            evaluate ( );
            reproduce ( );
            ++m_generation;
            Config::load ( );
            if ( config.save_population )
                save ( );
            if ( config.display_match )
                display ( );
            print_statistics ( );
        }
    }

    void print_fitness ( ) const noexcept {
        for ( auto const & i : m_population )
            std::wcout << L'<' << i.fitness << L' ' << i.age << L'>';
    }

    private:
    [[nodiscard]] static std::piecewise_linear_distribution<float> init_triangular_distribution ( ) noexcept {
        constexpr std::array<float, 3> i{ -1.0f, +0.0f, +1.0f }, w{ +0.0f, +1.0f, +0.0f };
        return std::piecewise_linear_distribution<float> ( i.begin ( ), i.end ( ), w.begin ( ) );
    }

    [[nodiscard]] static std::piecewise_linear_distribution<float> const & triangular_distribution ( ) noexcept {
        static std::piecewise_linear_distribution<float> const dis = init_triangular_distribution ( );
        return dis;
    }

    [[nodiscard]] static int sample ( ) noexcept { return uniformly_decreasing_discrete_distribution<BreedSize>{}( Rng::gen ( ) ); }
    [[nodiscard]] static std::tuple<int, int> sample_match ( ) noexcept {
        auto g = [] ( ) noexcept { return uniformly_decreasing_discrete_distribution<BreedSize>{}( Rng::gen ( ) ); };
        std::tuple<int, int> r = { g ( ), g ( ) };
        while ( std::get<0> ( r ) == std::get<1> ( r ) )
            std::get<1> ( r ) = g ( );
        return r;
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
                                                            [] ( Individual const & i ) noexcept { return i.age; } ) ) /
               static_cast<float> ( BreedSize );
    }

    friend class cereal::access;

    template<class Archive>
    void save ( Archive & ar_ ) const {
        constexpr int const ps = PopSize, fs = FieldSize, ni = NumInput, nn = NumNeurons, no = NumOutput;
        ar_ ( ps );
        ar_ ( fs );
        ar_ ( ni );
        ar_ ( nn );
        ar_ ( no );
        ar_ ( m_population );
        ar_ ( m_generation );
    }

    template<class Archive>
    void load ( Archive & ar_ ) {
        int ps = 0, fs = 0, ni = 0, nn = 0, no = 0;
        ar_ ( ps );
        ar_ ( fs );
        ar_ ( ni );
        ar_ ( nn );
        ar_ ( no );
        if ( ps != PopSize or fs != FieldSize or ni != NumInput or nn != NumNeurons or no != NumOutput ) {
            cls ( );
            std::wcout << L"parameters do not fit. <" << ps << L", " << fs << L", " << ni << L", " << nn << L", " << no << '>'
                       << nl;
            std::wcout << L"population size " << ps << nl;
            std::wcout << L"field size " << fs << nl;
            std::wcout << L"input size " << ni << nl;
            std::wcout << L"neurons " << nn << nl;
            std::wcout << L"output size " << no << nl;
            std::exit ( EXIT_SUCCESS );
        }
        ar_ ( m_population );
        ar_ ( m_generation );
    }

    void load ( ) noexcept { load_from_file_bin ( *this, "z://tmp", "population" ); }
    void save ( ) const noexcept { save_to_file_bin ( *this, "z://tmp", "population" ); }

    std::vector<Individual> m_population{ PopSize };
    int m_generation = 0;
};
