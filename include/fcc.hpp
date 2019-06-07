
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

#include <mkl.h>

#include <algorithm>
#include <array>
#include <limits>
#include <random>
#include <sax/iostream.hpp>

#include "rng.hpp"

// Space to be used for feed-forward-calculation.
template<int NumInput, int NumNeurons, int NumOutput>
struct InputBiasOutput {

    static_assert ( NumNeurons >= NumOutput, "number of neurons needs to be equal or larger than the number of required outputs" );

    static constexpr int NumBias    = 1;
    static constexpr int NumIns     = NumInput + NumBias;
    static constexpr int NumInsOuts = NumIns + NumNeurons;
    static constexpr int NumWeights = ( NumNeurons * ( 2 * NumInput + NumBias + NumNeurons ) ) / 2;

    using ibo_type = std::array<float, NumInsOuts>;

    InputBiasOutput ( ) noexcept { m_data[ NumInput ] = 1.0f; }

    constexpr float & operator[] ( int i ) noexcept { return m_data[ i ]; }
    constexpr float const & operator[] ( int i ) const noexcept { return m_data[ i ]; }

    [[nodiscard]] constexpr float * data ( ) noexcept { return m_data.data ( ); }
    [[nodiscard]] constexpr float const * data ( ) const noexcept { return m_data.data ( ); }

    [[nodiscard]] constexpr std::span<float> input ( ) noexcept { return { data ( ), NumInput }; }
    [[nodiscard]] constexpr std::span<float const> input ( ) const noexcept { return { data ( ), NumInput }; }

    template<typename Stream>
    [[maybe_unused]] friend Stream & operator<< ( Stream & out_, ibo_type const & d_ ) noexcept {
        for ( auto const v : d_.m_data )
            out_ << v << ' ';
        out_ << nl;
        return out_;
    }

    ibo_type m_data;
};

// A fully connected feed-forward cascade network.
template<int NumInput, int NumNeurons, int NumOutput>
struct FullyConnectedNeuralNetwork {

    static_assert ( NumNeurons >= NumOutput, "number of neurons needs to be equal or larger than the number of required outputs" );

    static constexpr int NumBias    = 1;
    static constexpr int NumIns     = NumInput + NumBias;
    static constexpr int NumInsOuts = NumIns + NumNeurons;
    static constexpr int NumWeights = ( NumNeurons * ( 2 * NumInput + NumBias + NumNeurons ) ) / 2;

    using ibo_type = InputBiasOutput<NumInput, NumNeurons, NumOutput>;
    using wgt_type = std::array<float, NumWeights>;

    using pointer        = typename wgt_type::pointer;
    using const_pointer  = typename wgt_type::const_pointer;
    using iterator       = typename wgt_type::iterator;
    using const_iterator = typename wgt_type::const_iterator;

    FullyConnectedNeuralNetwork ( ) noexcept {
        std::generate ( std::begin ( m_weights ), std::end ( m_weights ),
                        [] ( ) { return std::uniform_real_distribution<float> ( -1.0f, 1.0f ) ( Rng::gen ( ) ); } );
    }

    [[nodiscard]] const_pointer feed_forward ( pointer const ibo_ ) const noexcept {
        const_pointer wgt = m_weights.data ( );
        for ( int i = NumIns; i < NumInsOuts; wgt += i++ )
            ibo_[ i ] = activation_elliotsig ( cblas_sdot ( i, ibo_, 1, wgt, 1 ), 1.0f );
        return ibo_ + NumInsOuts - NumOutput;
    }

    [[nodiscard]] inline float activation_bipolar ( float net_, float const alpha_ ) const noexcept {
        net_ *= alpha_;
        return 2.0f / ( 1.0f + std::exp ( -2.0f * net_ ) ) - 1.0f;
    }
    [[nodiscard]] inline float activation_elliotsig ( float net_, float const alpha_ ) const noexcept {
        net_ *= alpha_;
        return net_ / ( 1.0f + std::abs ( net_ ) );
    }

    template<typename Stream>
    [[maybe_unused]] friend Stream & operator<< ( Stream & out_, FullyConnectedNeuralNetwork const & w_ ) noexcept {
        for ( auto const v : w_.m_weights )
            out_ << v << ' ';
        out_ << nl;
        return out_;
    }

    [[nodiscard]] float run ( ) const noexcept {
        return std::uniform_real_distribution<float> ( 0.0f, 10'000.0f ) ( Rng::gen ( ) );
    }

    [[nodiscard]] float & operator[] ( int i_ ) noexcept { return m_weights[ i_ ]; }
    [[nodiscard]] float const & operator[] ( int i_ ) const noexcept { return m_weights[ i_ ]; }

    [[nodiscard]] constexpr pointer data ( ) noexcept { return m_weights.data ( ); }
    [[nodiscard]] constexpr const_pointer * data ( ) const noexcept { return m_weights.data ( ); }

    [[nodiscard]] iterator begin ( ) noexcept { return iterator ( m_weights.begin ( ) ); }
    [[nodiscard]] iterator end ( ) noexcept { return iterator ( m_weights.end ( ) ); }
    [[nodiscard]] const_iterator begin ( ) const noexcept { return const_iterator ( m_weights.begin ( ) ); }
    [[nodiscard]] const_iterator cbegin ( ) const noexcept { return const_iterator ( m_weights.cbegin ( ) ); }
    [[nodiscard]] const_iterator end ( ) const noexcept { return const_iterator ( m_weights.end ( ) ); }
    [[nodiscard]] const_iterator cend ( ) const noexcept { return const_iterator ( m_weights.cend ( ) ); }

    private:
    wgt_type m_weights;
};
