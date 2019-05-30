
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

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <array>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <list>
#include <map>
#include <random>
#include <sax/iostream.hpp>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <mkl.h>

#include <xsimd/xsimd.hpp>

#include "rng.hpp"

namespace xs = xsimd;

template<int NumInput, int NumNeurons, int NumOutput>
struct FullyConnectedNeuralNetwork {

    static_assert ( NumNeurons >= NumOutput, "number of neurons needs to be equal or larger than the number of required outputs" );

    static constexpr int NumBias    = 1;
    static constexpr int NumIns     = NumInput + NumBias;
    static constexpr int NumInsOuts = NumIns + NumNeurons;
    static constexpr int NumWeights = ( NumNeurons * ( 2 * NumInput + NumBias + NumNeurons ) ) / 2;

    std::array<float, NumInsOuts> m_input_bias_output{};
    std::array<float, NumWeights> m_weights;

    FullyConnectedNeuralNetwork ( ) noexcept {
        m_input_bias_output[ NumInput ] = 1.0f;
        std::generate ( std::begin ( m_weights ), std::end ( m_weights ),
                        [] ( ) { return std::uniform_real_distribution<float> ( -1.0f, 1.0f ) ( Rng::gen ( ) ); } );
        print_ibo ( );
    }

    template<typename... Args>
    void input ( Args &&... args_ ) noexcept {
        static_assert ( sizeof...( args_ ) == NumInput, "input has to be equal to NumInput" );
        int i = 0; // starting index, can be an argument.
        ( ( m_input_bias_output[ i++ ] = args_ ), ... );
        print_ibo ( );
    }

    void feedForward ( ) noexcept {
        float * wgt = m_weights.data ( );
        for ( int col = NumIns; col < NumInsOuts; wgt += col++ ) {
            m_input_bias_output[ col ] = activation_bipolar ( cblas_sdot ( col, m_input_bias_output.data ( ), 1, wgt, 1 ), 1.0f );
            print_ibo ( );
        }
    }

    [[nodiscard]] inline float activation_bipolar ( const float net_, const float alpha_ ) noexcept {
        return 2.0f / ( 1.0f + std::exp ( -2.0f * net_ * alpha_ ) ) - 1.0f;
    }

    void print_ibo ( ) noexcept {
        for ( const auto v : m_input_bias_output )
            std::cout << v << ' ';
        std::cout << nl;
    }
};

int main ( ) {

    FullyConnectedNeuralNetwork<4, 5, 3> nn;

    nn.input ( 0.1f, -0.2f, 0.12f, -0.6f );
    nn.feedForward ( );

    return EXIT_SUCCESS;
}
