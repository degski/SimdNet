
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

#include "fcc.hpp"
#include "rng.hpp"

template<int PopSize, int NumInput, int NumNeurons, int NumOutput>
struct Population {

    static constexpr int BreedSize = PopSize / 2;

    using Network     = FullyConnectedNeuralNetwork<NumInput, NumNeurons, NumOutput>;
    using SampleTable = std::array<float, BreedSize>;

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
                        [] ( Individual & i ) { i.id = new Network ( ); } );
    }

    ~Population ( ) noexcept {
        std::for_each ( std::execution::par_unseq, std::begin ( m_population ), std::end ( m_population ),
                        [] ( Individual & i ) { delete i.id; } );
    }

    void evaluate ( ) noexcept {
        std::for_each ( std::execution::par_unseq, std::begin ( m_population ), std::end ( m_population ), [] ( Individual & i ) {
            i.fitness = i.id->run ( );
            i.age += 1;
        } );
        std::sort ( std::begin ( m_population ), std::end ( m_population ),
                    [] ( Individual const & a, Individual const & b ) { return a.fitness > b.fitness; } );
    }

    void reproduce ( ) noexcept {
        std::cout << nl;
        std::cout << nl;
        for ( auto & f : m_sample_table )
            std::cout << f << ' ';
        std::cout << nl;
    }

    [[nodiscard]] static constexpr SampleTable generate_sample_table ( ) noexcept {
        SampleTable table{};
        for ( int n = BreedSize, i = 0, c = n; i < BreedSize; ++i, c += --n )
            table[ i ] = static_cast<float> ( static_cast<double> ( c ) /
                                              ( static_cast<double> ( BreedSize ) * static_cast<double> ( BreedSize + 1 ) * 0.5 ) );
        return table;
    }

    static constexpr SampleTable m_sample_table = generate_sample_table ( );
};

template<int NumInput, int NumNeurons, int NumOutput>
void crossover ( FullyConnectedNeuralNetwork<NumInput, NumNeurons, NumOutput> & p0_,
                 FullyConnectedNeuralNetwork<NumInput, NumNeurons, NumOutput> & p1_ ) noexcept {
    int const cop = std::uniform_int_distribution<int> (
        0, FullyConnectedNeuralNetwork<NumInput, NumNeurons, NumOutput>::NumWeights - 2 ) ( Rng::gen ( ) ); // crossover point.
    // Swap the smallest range of the two (separated by the cossover point).
    if ( cop < FullyConnectedNeuralNetwork<NumInput, NumNeurons, NumOutput>::NumWeights / 2 )
        std::swap_ranges ( p0_.weights ( ), p0_.weights ( ) + cop, p1_.weights ( ) );
    else
        std::swap_ranges ( p0_.weights ( ) + cop,
                           p0_.weights ( ) + FullyConnectedNeuralNetwork<NumInput, NumNeurons, NumOutput>::NumWeights,
                           p1_.weights ( ) + cop );
}

template<int NumInput, int NumNeurons, int NumOutput>
void mutate ( FullyConnectedNeuralNetwork<NumInput, NumNeurons, NumOutput> & p_ ) noexcept {
    int const mup = std::uniform_int_distribution<int> (
        0, FullyConnectedNeuralNetwork<NumInput, NumNeurons, NumOutput>::NumWeights - 1 ) ( Rng::gen ( ) ); // mutation point.
    p_.weights[ mup ] = std::normal_distribution<float> ( 0.0f, 1.0f ) ( Rng::gen ( ) );
}
