
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

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include <limits>
#include <random>

#include <sax/prng.hpp>
#include <sax/uniform_int_distribution.hpp>

#if defined( _DEBUG )
#    define RANDOM 0
#else
#    define RANDOM 1
#endif

struct Rng final {

    Rng ( Rng && )      = delete;
    Rng ( const Rng & ) = delete;

    Rng & operator= ( Rng && ) = delete;
    Rng & operator= ( const Rng & ) = delete;

    // A pareto-variate, the defaults give the 'famous' 80/20 distribution.
    template<typename T = float>
    [[nodiscard]] static T pareto_variate ( const T min_   = T{ 1 },
                                            const T alpha_ = { std::log ( T{ 5 } ) / std::log ( T{ 4 } ) } ) noexcept {
        assert ( min_ > T{ 0 } );
        assert ( alpha_ > T{ 0 } );
        static std::uniform_real_distribution<T> dis ( std::numeric_limits<T>::min ( ), T{ 1 } );
        return min_ / std::pow ( dis ( Rng::gen ( ) ), T{ 1 } / alpha_ );
    }

    [[nodiscard]] static bool bernoulli ( double p_ = 0.5 ) noexcept { return std::bernoulli_distribution ( p_ ) ( Rng::gen ( ) ); }

    static void seed ( const std::uint64_t s_ = 0u ) noexcept { Rng::gen ( ).seed ( s_ ? s_ : sax::os_seed ( ) ); }

    [[nodiscard]] static sax::Rng & gen ( ) noexcept {
        static thread_local sax::Rng generator ( RANDOM ? sax::os_seed ( ) : sax::fixed_seed ( ) );
        return generator;
    }
};

#undef RANDOM
