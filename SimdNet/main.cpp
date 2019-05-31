
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

#include <plf/plf_nanotimer.h>

#include <sax/uniform_int_distribution.hpp>

#include "../../multi_array/include/multi_array.hpp"
#include "fcc.hpp"

template<int S>
struct SnakeSpace {

    static_assert ( S % 2 != 0, "uneven size only" );

    static constexpr int Base = -( S / 2 );
    static constexpr int Size = S;

    enum class Direction : char { ne = 1, ea, se, so, sw, we, nw, no };
    enum class Object : char { none = 0, snake, food };

    void new_food ( ) noexcept {
        auto idx = [ this ]( ) { return m_dis ( Rng::gen ( ) ); };
        int x = idx ( ), y = idx ( );
        while ( Object::none != m_field.at ( x, y ) )
            x = idx ( ), y = idx ( );
        m_field.at ( x, y ) = Object::food;
    }

    sax::Matrix<Object, Size, Size, Base, Base> m_field;
    sax::uniform_int_distribution<int> m_dis{ Base, -Base };
};

int main ( ) {

    SnakeSpace<39> ss;

    constexpr int in = 128;

    FullyConnectedNeuralNetwork<in, 150, 1> nn;

    typename FullyConnectedNeuralNetwork<in, 150, 1>::ibo_type ibo;

    float x = 0.0;

    plf::nanotimer t;

    t.start ( );

    for ( int i = 0; i < 100'000; ++i ) {
        for ( int j = 0; j < in; ++j )
            ibo[ j ] = std::uniform_real_distribution<float> ( -0.25f, 0.25f ) ( Rng::gen ( ) );
        auto out = nn.feed_forward ( ibo.data ( ) );
        x += out[ 0 ];
    }

    std::cout << ( std::uint64_t ) t.get_elapsed_ms ( ) << nl;

    std::cout << x << nl;

    return EXIT_SUCCESS;
}

/*


template<int NumInput, int NumNeurons, int NumOutput>
struct FullyConnectedNeuralNetwork {

    static_assert ( NumNeurons >= NumOutput, "number of neurons needs to be equal or larger than the number of required outputs" );

    static constexpr int NumBias    = 1;
    static constexpr int NumIns     = NumInput + NumBias;
    static constexpr int NumInsOuts = NumIns + NumNeurons;
    static constexpr int NumWeights = ( NumNeurons * ( 2 * NumInput + NumBias + NumNeurons ) ) / 2;

    using ibo_type = std::array<float, NumInsOuts>;
    using wgt_type = std::array<float, NumWeights>;

    ibo_type m_input_bias_output;
    wgt_type m_weights;

    FullyConnectedNeuralNetwork ( ) noexcept {
        m_input_bias_output[ NumInput ] = 1.0f;
        std::generate ( std::begin ( m_weights ), std::end ( m_weights ),
                        [] ( ) { return std::uniform_real_distribution<float> ( -1.0f, 1.0f ) ( Rng::gen ( ) ); } );
    }

    template<typename... Args>
    void input ( Args &&... args_ ) noexcept {
        static_assert ( sizeof...( args_ ) == NumInput, "input has to be equal to NumInput" );
        int i = 0;
        ( ( m_input_bias_output[ i++ ] = args_ ), ... );
    }

    [[nodiscard]] constexpr std::span<float> output ( ) noexcept {
        return { m_input_bias_output.data ( ) + NumInsOuts - NumOutput, NumOutput };
    }

    void feed_forward ( ) noexcept {
        float * ibo = m_input_bias_output.data ( ), * wgt = m_weights.data ( );
        for ( int i = NumIns; i < NumInsOuts; wgt += i++ )
            ibo[ i ] = activation_bipolar ( cblas_sdot ( i, ibo, 1, wgt, 1 ), 1.0f );
    }

    template<typename... Args>
    [[nodiscard]] std::span<float> feed_forward ( Args &&... args_ ) noexcept {
        float * ibo = m_input_bias_output.data ( );
        ( ( *ibo++ = args_ ), ... );
        ibo = m_input_bias_output.data ( );
        float *wgt = m_weights.data ( );
        for ( int i = NumIns; i < NumInsOuts; wgt += i++ )
            ibo[ i ] = activation_bipolar ( cblas_sdot ( i, ibo, 1, wgt, 1 ), 1.0f );
        return { m_input_bias_output.data ( ) + NumInsOuts - NumOutput, NumOutput };
    }

    [[nodiscard]] std::span<float> feed_forward ( float * const ibo_ ) noexcept {
        float * wgt = m_weights.data ( );
        for ( int i = NumIns; i < NumInsOuts; wgt += i++ )
            ibo_[ i ] = activation_elliotsig ( cblas_sdot ( i, ibo_, 1, wgt, 1 ), 1.0f );
        return { ibo_ + NumInsOuts - NumOutput, NumOutput };
    }

    [[nodiscard]] inline float activation_bipolar ( float net_, float const alpha_ ) noexcept {
        net_ *= alpha_;
        return 2.0f / ( 1.0f + std::exp ( -2.0f * net_ ) ) - 1.0f;
    }
    [[nodiscard]] inline float activation_elliotsig ( float net_, float const alpha_ ) noexcept {
        net_ *= alpha_;
        return net_ / ( 1.0f + std::abs ( net_ ) );
    }

    void print_ibo ( ) noexcept {
        for ( const auto v : m_input_bias_output )
            std::cout << v << ' ';
        std::cout << nl;
    }
};

*/
