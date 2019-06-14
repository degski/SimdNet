
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
#include <array>

#include <sax/uniform_int_distribution.hpp>

template<int Size, typename T = int>
struct uniformly_decreasing_discrete_distribution;

template<int Size, typename T = int>
struct param_type {

    static_assert ( Size > 1, "size should be larger than 1" );

    template<int Size, typename T>
    friend struct uniformly_decreasing_discrete_distribution;

    using result_type = T;
    using distribution_type = uniformly_decreasing_discrete_distribution<Size, T>;

    constexpr param_type ( ) noexcept = default;

    [[nodiscard]] std::vector<double> probabilities ( ) const {
        std::vector<double> table{ Size, 0.0 };
        for ( T n = Size, i = 0; i < Size; ++i, --n )
            table[ i ] = static_cast<double> ( n ) / static_cast<double> ( Sum );
        return table;
    }

    [[nodiscard]] bool operator== ( const param_type & right ) const noexcept { return true; };
    [[nodiscard]] bool operator!= ( const param_type & right ) const noexcept { return false; };

    private:

    using SampleTable = std::array<T, Size>;

    static constexpr int Sum = Size % 2 == 0 ? ( ( Size / 2 ) * ( Size + 1 ) ) : ( Size * ( ( Size + 1 ) / 2 ) );

    [[nodiscard]] static constexpr SampleTable generate_sample_table ( ) noexcept {
        SampleTable table{};
        for ( T n = Size, i = 0, c = n; i < Size; ++i, c += --n )
            table[ i ] = c;
        return table;
    }

    static constexpr SampleTable const m_sample_table = generate_sample_table ( );
};

template<int Size, typename T>
struct uniformly_decreasing_discrete_distribution : param_type<Size, T> {

    using param_type = param_type<Size, T>;
    using result_type = typename param_type::result_type;

    // Sample with a linearly decreasing probability.
    // Iff size was 3, the probabilities of the CDF would
    // be 3/6, 5/6, 6/6 (or 3/6, 2/6, 1/6 for the PDF).
    template<typename Generator>
    [[nodiscard]] result_type operator( ) ( Generator & gen_ ) noexcept {
        int const i = sax::uniform_int_distribution<int> ( 0, param_type::Sum ) ( gen_ ); // needs uniform bits generator.
        return static_cast<result_type> (
            std::lower_bound ( std::begin ( param_type::m_sample_table ), std::end ( param_type::m_sample_table ), i ) -
            std::begin ( param_type::m_sample_table ) );
    }

    void reset ( ) const noexcept {}

    [[nodiscard]] constexpr result_type min ( ) const noexcept { return result_type{ 0 }; }
    [[nodiscard]] constexpr result_type max ( ) const noexcept { return result_type{ Size - 1 }; }
};
