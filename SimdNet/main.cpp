
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
#include <cstring>

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

#include "population.hpp"

int main5658 ( ) {

    Population<4'096 * 8, 39, 24, 5, 4> p;

    p.run ( );

    return EXIT_SUCCESS;
}

#ifdef small
#    define org_small small
#    undef small
#endif

// http://www.keithschwarz.com/darts-dice-coins/

// The probability and alias tables.
template<typename T = int, typename U = float>
struct VoseTables {
    std::vector<U> m_probability{};
    std::vector<T> m_alias{};
    explicit VoseTables ( std::size_t const n_ ) :
        m_probability ( n_, U{ 0 } ), m_alias ( n_, T{ 0 } ) {}

    [[nodiscard]] int size ( ) const noexcept { return static_cast<int> ( m_probability.size ( ) ); }
};

template<typename U>
[[nodiscard]] inline U pop ( std::vector<U> & v_ ) noexcept {
    U const r = v_.back ( );
    v_.pop_back ( );
    return r;
}

template<typename T = int, typename U = float>
VoseTables<T, U> init ( std::vector<U> const & probability_ ) noexcept {
    assert ( probability_.size ( ) > 0u );
    std::vector<U> probability{ probability_ };
    U const n_div_sum = static_cast<U> ( static_cast<double> ( probability.size ( ) ) /
                        std::reduce ( std::execution::par_unseq, std::begin ( probability ), std::end ( probability ), 0.0,
                                      []( double const a, double const b ) { return a + b; } ) );
    std::for_each ( std::execution::par_unseq, std::begin ( probability ), std::end ( probability ),
                    [n_div_sum]( U & v ) { return v *= n_div_sum; } );
    std::vector<T> large, small;
    large.reserve ( probability.size ( ) );
    small.reserve ( probability.size ( ) );
    T i = 0;
    for ( U const p : probability ) {
        if ( p >= U{ 1 } )
            large.push_back ( i );
        else
            small.push_back ( i );
        ++i;
    }
    VoseTables<T, U> tables ( probability.size ( ) );
    while ( large.size ( ) and small.size ( ) ) {
        T const g = pop ( large ), l = pop ( small );
        tables.m_probability[ l ] = probability[ l ];
        tables.m_alias[ l ]       = g;
        probability[ g ]      = ( ( probability[ g ] + probability[ l ] ) - U{ 1 } );
        if ( probability[ g ] >= U{ 1 } )
            large.push_back ( g );
        else
            small.push_back ( g );
    }
    while ( large.size ( ) )
        tables.m_probability[ pop ( large ) ] = U{ 1 };
    while ( small.size ( ) )
        tables.m_probability[ pop ( small ) ] = U{ 1 };
    return tables;
}

#ifdef org_small
#    define small org_small
#    undef org_small
#endif

template<typename T = int, typename U = float>
int next ( VoseTables<T, U> & dis_ ) {
    int const column = sax::uniform_int_distribution<int> ( 0, dis_.size ( ) - 1 ) ( Rng::gen ( ) );
    return Rng::bernoulli ( dis_.m_probability[ column ] ) ? column : dis_.m_alias[ column ];
}

int main ( ) {

    auto dis = init ( std::vector<float>{ 0.5f, 0.3f, 0.2f } );

    int buck[ 3 ]{};

    for ( int i = 0; i < 1'000'000; ++i )
        ++buck[ next ( dis ) ];

    for ( int i = 0; i < 3; ++i )
        std::cout << buck[ i ] << ' ';
    std::cout << nl;

    return EXIT_SUCCESS;
}

/*

    f(x) = 255 - a * log ( x + 1 ) with a = 255 / log ( 256 ) ~=~ 46


-fsanitize=address

C:\Program Files\LLVM\lib\clang\9.0.0\lib\windows\clang_rt.asan_cxx-x86_64.lib
C:\Program Files\LLVM\lib\clang\9.0.0\lib\windows\clang_rt.asan-preinit-x86_64.lib
C:\Program Files\LLVM\lib\clang\9.0.0\lib\windows\clang_rt.asan-x86_64.lib

*/
