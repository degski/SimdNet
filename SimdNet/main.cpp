
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

#include <boost/circular_buffer.hpp>

#include <plf/plf_nanotimer.h>

#include <sax/uniform_int_distribution.hpp>

#include "fcc.hpp"

struct Point {
    char x, y;

    [[nodiscard]] bool operator== ( Point const & rhs_ ) const noexcept { return rhs_.x == x and rhs_.y == y; }
    [[nodiscard]] bool operator!= ( Point const & rhs_ ) const noexcept { return not operator== ( rhs_ ); }

    template<typename Stream>
    [[maybe_unused]] friend Stream & operator<< ( Stream & out_, Point const & p_ ) noexcept {
        out_ << '<' << ( int ) p_.x << ' ' << ( int ) p_.y << '>';
        return out_;
    }
};

[[nodiscard]] Point operator+ ( Point && p1_, Point const & p2_ ) noexcept {
    p1_.x += p2_.x;
    p1_.y += p2_.y;
    return p1_;
}

[[nodiscard]] Point operator- ( Point const & p1_, Point const & p2_ ) noexcept {
    return { static_cast<char> ( p1_.x - p2_.x ), static_cast<char> ( p1_.y - p2_.y ) };
}

template<int B>
[[nodiscard]] Point random_point ( ) noexcept {
    auto idx = []( ) { return static_cast<char> ( sax::uniform_int_distribution<int>{ -B, B }( Rng::gen ( ) ) ); };
    return { idx ( ), idx ( ) };
}

template<int S>
struct SnakeSpace {

    static_assert ( S % 2 != 0, "uneven size only" );

    static constexpr int Base = S / 2;
    static constexpr int Size = S;

    enum class ScanDirection : int { no, ne, ea, se, so, sw, we, nw };
    enum class MoveDirection : int { no, ea, so, we };

    using SnakeBody = boost::circular_buffer<Point>;

    [[nodiscard]] inline bool in_range ( Point const & p_ ) const noexcept {
        return p_.x >= -Base and p_.y >= -Base and p_.x <= Base and p_.y <= Base;
    }

    [[nodiscard]] inline bool snake_body_contains ( Point const & p_ ) const noexcept {
        return std::find ( std::cbegin ( m_snake_body ), std::cend ( m_snake_body ), p_ ) != std::cend ( m_snake_body );
    }

    [[nodiscard]] inline bool valid_empty_point ( Point const & p_ ) const noexcept {
        return in_range ( p_ ) and not snake_body_contains ( p_ );
    }

    void random_food ( ) noexcept {
        Point f = random_point<Base> ( );
        while ( snake_body_contains ( f ) )
            f = random_point<Base> ( );
        m_food = f;
    }

    SnakeSpace ( ) noexcept { init ( ); }

    [[nodiscard]] Point extend_head ( ) const noexcept {
        switch ( m_direction ) {
            case MoveDirection::no: return Point{ +0, +1 } + m_snake_body.front ( );
            case MoveDirection::ea: return Point{ +1, +0 } + m_snake_body.front ( );
            case MoveDirection::so: return Point{ +0, -1 } + m_snake_body.front ( );
            case MoveDirection::we: return Point{ -1, +0 } + m_snake_body.front ( );
        }
        return { 0, 0 };
    }

    void move ( ) noexcept {
        ++m_move_count;
        --m_energy;
        m_snake_body.push_front ( extend_head ( ) );
        if ( not m_energy or not in_range ( m_snake_body.front ( ) ) ) {
            std::wcout << L"the grim reaper has collected his reward" << nl;
            std::exit ( EXIT_SUCCESS );
        }
        else if ( m_snake_body.front ( ) != m_food ) {
            m_snake_body.pop_back ( );
        }
        else {
            m_energy += 50;
            random_food ( );
        }
    }

    void turn_right ( ) noexcept { m_direction = static_cast<MoveDirection> ( ( static_cast<int> ( m_direction ) + 3 ) % 4 ); }
    void turn_left ( ) noexcept { m_direction = static_cast<MoveDirection> ( ( static_cast<int> ( m_direction ) + 1 ) % 4 ); }

    void init ( ) noexcept {
        m_direction  = static_cast<MoveDirection> ( sax::uniform_int_distribution<int>{ 0, 3 }( Rng::gen ( ) ) );
        m_move_count = 0;
        m_energy     = 100;
        m_snake_body.push_front ( random_point<Base - 6> ( ) ); // the new tail.
        m_snake_body.push_front ( extend_head ( ) );
        m_snake_body.push_front ( extend_head ( ) ); // the new head.
        random_food ( );
    }

    void run ( ) noexcept {
        for ( int i = 0; i < 100; ++i ) {
            print ( );
            std::wcout << nl;
            move ( );
        }
    }

    // Input (activation) for distance to wall.
    [[nodiscard]] static float distance_to_wall ( Point const & hp_, ScanDirection const & dir_ ) noexcept {
        switch ( dir_ ) {
            case ScanDirection::no: return 1.0f / ( Base - hp_.y + 1 );
            case ScanDirection::ne: return 1.0f / ( 2 * std::min ( Base - hp_.x, Base - hp_.y ) + 1 );
            case ScanDirection::ea: return 1.0f / ( Base - hp_.x + 1 );
            case ScanDirection::se: return 1.0f / ( 2 * std::min ( Base - hp_.x, Base + hp_.y ) + 1 );
            case ScanDirection::so: return 1.0f / ( Base + hp_.y + 1 );
            case ScanDirection::sw: return 1.0f / ( 2 * std::min ( Base + hp_.x, Base + hp_.y ) + 1 );
            case ScanDirection::we: return 1.0f / ( Base + hp_.x + 1 );
            case ScanDirection::nw: return 1.0f / ( 2 * std::min ( Base + hp_.x, Base - hp_.y ) + 1 );
        }
        return NAN;
    }

    // Input (activation) for distance to point.
    [[nodiscard]] static float distance_point_to_point ( Point const & p0_, Point const & p1_,
                                                         ScanDirection const & dir_ ) noexcept {
        switch ( dir_ ) {
            case ScanDirection::no: return p0_.x != p1_.x or p0_.y >= p1_.y ? 0.0f : 1.0f / ( p1_.y - p0_.y );
            case ScanDirection::ne:
                return p0_.x >= p1_.x or p0_.y >= p1_.y or ( ( p0_.x - p0_.y ) != ( p1_.x - p1_.y ) )
                           ? 0.0f
                           : 1.0f / ( ( p1_.x - p0_.x ) + ( p1_.y - p0_.y ) );
            case ScanDirection::ea: return p0_.y != p1_.y or p0_.x >= p1_.x ? 0.0f : 1.0f / ( p1_.x - p0_.x );
            case ScanDirection::se:
                return p0_.x >= p1_.x or p0_.y <= p1_.y or ( ( p0_.x + p0_.y ) != ( p1_.x + p1_.y ) )
                           ? 0.0f
                           : 1.0f / ( ( p1_.x - p0_.x ) + ( p0_.y - p1_.y ) );
            case ScanDirection::so: return p0_.x != p1_.x or p0_.y <= p1_.y ? 0.0f : 1.0f / ( p0_.y - p1_.y );
            case ScanDirection::sw:
                return p0_.x <= p1_.x or p0_.y <= p1_.y or ( ( p0_.x - p0_.y ) != ( p1_.x - p1_.y ) )
                           ? 0.0f
                           : 1.0f / ( ( p0_.x - p1_.x ) + ( p0_.y - p1_.y ) );
            case ScanDirection::we: return p0_.y != p1_.y or p0_.x <= p1_.x ? 0.0f : 1.0f / ( p0_.x - p1_.x );
            case ScanDirection::nw:
                return p0_.x <= p1_.x or p0_.y >= p1_.y or ( ( p0_.x + p0_.y ) != ( p1_.x + p1_.y ) )
                           ? 0.0f
                           : 1.0f / ( ( p0_.x - p1_.x ) + ( p1_.y - p0_.y ) );
        }
        return NAN;
    }

    static void distances_point_to_point ( float * out_, Point const & p0_, Point const & p1_ ) noexcept {
        out_[ 0 ] = p0_.x != p1_.x or p0_.y >= p1_.y ? 0.0f : 1.0f / ( p1_.y - p0_.y );
        out_[ 1 ] = p0_.x >= p1_.x or p0_.y >= p1_.y or ( ( p0_.x - p0_.y ) != ( p1_.x - p1_.y ) )
                        ? 0.0f
                        : 1.0f / ( ( p1_.x - p0_.x ) + ( p1_.y - p0_.y ) );
        out_[ 2 ] = p0_.y != p1_.y or p0_.x >= p1_.x ? 0.0f : 1.0f / ( p1_.x - p0_.x );
        out_[ 3 ] = p0_.x >= p1_.x or p0_.y <= p1_.y or ( ( p0_.x + p0_.y ) != ( p1_.x + p1_.y ) )
                        ? 0.0f
                        : 1.0f / ( ( p1_.x - p0_.x ) + ( p0_.y - p1_.y ) );
        out_[ 4 ] = p0_.x != p1_.x or p0_.y <= p1_.y ? 0.0f : 1.0f / ( p0_.y - p1_.y );
        out_[ 5 ] = p0_.x <= p1_.x or p0_.y <= p1_.y or ( ( p0_.x - p0_.y ) != ( p1_.x - p1_.y ) )
                        ? 0.0f
                        : 1.0f / ( ( p0_.x - p1_.x ) + ( p0_.y - p1_.y ) );
        out_[ 6 ] = p0_.y != p1_.y or p0_.x <= p1_.x ? 0.0f : 1.0f / ( p0_.x - p1_.x );
        out_[ 7 ] = p0_.x <= p1_.x or p0_.y >= p1_.y or ( ( p0_.x + p0_.y ) != ( p1_.x + p1_.y ) )
                        ? 0.0f
                        : 1.0f / ( ( p0_.x - p1_.x ) + ( p1_.y - p0_.y ) );
    }

    static void distances_point_to_point2 ( float * out_, Point const & p0_, Point const & p1_ ) noexcept {
        std::memset ( out_, 0, 8 * sizeof ( float ) );
        Point const s = p0_ - p1_;
        if ( 0 == s.x ) {
            if ( s.y > 0 )
                out_[ 0 ] = 1.0f / s.y;
            else
                out_[ 4 ] = 1.0f / s.y;
        }
    }

    [[nodiscard]] float distance_to_body ( ScanDirection const & dir_ ) const noexcept {
        Point const head = m_snake_body.front ( );
        float distance   = 0.0f;
        auto const end   = std::end ( m_snake_body );
        auto it          = std::next ( std::begin ( m_snake_body ) );
        while ( it != end ) {
            float const d = distance_point_to_point ( head, *it, dir_ );
            if ( d > distance )
                distance = d;
            ++it;
        }
        return distance;
    }

    void distances_to_body ( float * dist_ ) const noexcept {
        Point const & head = m_snake_body.front ( );
        auto const end     = std::end ( m_snake_body );
        auto it            = std::next ( std::begin ( m_snake_body ) ); // This assumes the length of the snake is at least 2.
        float d[ 8 ];
        distances_point_to_point ( dist_, head, *it );
        while ( ++it != end ) {
            distances_point_to_point ( d, head, *it );
            if ( d[ 0 ] > dist_[ 0 ] )
                dist_[ 0 ] = d[ 0 ];
            if ( d[ 1 ] > dist_[ 1 ] )
                dist_[ 1 ] = d[ 1 ];
            if ( d[ 2 ] > dist_[ 2 ] )
                dist_[ 2 ] = d[ 2 ];
            if ( d[ 3 ] > dist_[ 3 ] )
                dist_[ 3 ] = d[ 3 ];
            if ( d[ 4 ] > dist_[ 4 ] )
                dist_[ 4 ] = d[ 4 ];
            if ( d[ 5 ] > dist_[ 5 ] )
                dist_[ 5 ] = d[ 5 ];
            if ( d[ 6 ] > dist_[ 6 ] )
                dist_[ 6 ] = d[ 6 ];
            if ( d[ 7 ] > dist_[ 7 ] )
                dist_[ 7 ] = d[ 7 ];
        }
    }

    void print ( ) const noexcept {
        for ( int y = -Base; y <= Base; ++y ) {
            for ( int x = -Base; x <= Base; ++x ) {
                Point const p{ static_cast<char> ( x ), static_cast<char> ( y ) };
                if ( p == m_food )
                    std::wcout << L" o ";
                else if ( snake_body_contains ( p ) ) {
                    if ( p == m_snake_body.front ( ) )
                        std::wcout << L" x ";
                    else
                        std::wcout << L" s ";
                }
                else
                    std::wcout << L" . ";
            }
            std::wcout << nl;
        }
    }

    MoveDirection m_direction;
    int m_move_count, m_energy;
    SnakeBody m_snake_body{ 1'024 };
    Point m_food;
};

void distances_point_to_point ( float * out_, Point const & p0_, Point const & p1_ ) noexcept {
    out_[ 0 ] = p0_.x != p1_.x or p0_.y >= p1_.y ? 0.0f : 1.0f / ( p1_.y - p0_.y );
    out_[ 1 ] = p0_.x >= p1_.x or p0_.y >= p1_.y or ( ( p0_.x - p0_.y ) != ( p1_.x - p1_.y ) )
                    ? 0.0f
                    : 1.0f / ( ( p1_.x - p0_.x ) + ( p1_.y - p0_.y ) );
    out_[ 2 ] = p0_.y != p1_.y or p0_.x >= p1_.x ? 0.0f : 1.0f / ( p1_.x - p0_.x );
    out_[ 3 ] = p0_.x >= p1_.x or p0_.y <= p1_.y or ( ( p0_.x + p0_.y ) != ( p1_.x + p1_.y ) )
                    ? 0.0f
                    : 1.0f / ( ( p1_.x - p0_.x ) + ( p0_.y - p1_.y ) );
    out_[ 4 ] = p0_.x != p1_.x or p0_.y <= p1_.y ? 0.0f : 1.0f / ( p0_.y - p1_.y );
    out_[ 5 ] = p0_.x <= p1_.x or p0_.y <= p1_.y or ( ( p0_.x - p0_.y ) != ( p1_.x - p1_.y ) )
                    ? 0.0f
                    : 1.0f / ( ( p0_.x - p1_.x ) + ( p0_.y - p1_.y ) );
    out_[ 6 ] = p0_.y != p1_.y or p0_.x <= p1_.x ? 0.0f : 1.0f / ( p0_.x - p1_.x );
    out_[ 7 ] = p0_.x <= p1_.x or p0_.y >= p1_.y or ( ( p0_.x + p0_.y ) != ( p1_.x + p1_.y ) )
                    ? 0.0f
                    : 1.0f / ( ( p0_.x - p1_.x ) + ( p1_.y - p0_.y ) );
}

void dp2p ( float * out_, Point const & p0_, Point const & p1_ ) noexcept {
    Point const s = p0_ - p1_;
    if ( 0 == s.x ) {
        if ( s.y < 0 )
            out_[ 0 ] = 1.0f / -s.y;
        else
            out_[ 4 ] = 1.0f / +s.y;
        return;
    }
    if ( s.x == s.y ) {
        if ( s.x < 0 )
            out_[ 1 ] = 0.5f / -s.x;
        else
            out_[ 5 ] = 0.5f / +s.x;
        return;
    }
    if ( 0 == s.y ) {
        if ( s.x < 0 )
            out_[ 2 ] = 1.0f / -s.x;
        else
            out_[ 6 ] = 1.0f / +s.x;
        return;
    }
    if ( s.x == -s.y ) {
        if ( s.x < 0 )
            out_[ 3 ] = 0.5f / -s.x;
        else
            out_[ 7 ] = 0.5f / +s.x;
        return;
    }
}

int main ( ) {

    Point b{ 8, -7 };
    Point h{ -2, 1 };

    float dists1[ 8 ]{};

    dp2p ( dists1, b, h );

    for ( auto f : dists1 )
        std::cout << f << ' ';
    std::cout << nl;

    float dists2[ 8 ]{};

    distances_point_to_point ( dists2, b, h );

    for ( auto f : dists2 )
        std::cout << f << ' ';
    std::cout << nl;

    /*

    SnakeSpace<17> ss;

    Point f{ -8, -8 };
    Point p{ 6, 6 };

    std::cout << SnakeSpace<17>::distance_point_to_point ( p, f, SnakeSpace<17>::ScanDirection::no ) << nl;
    std::cout << SnakeSpace<17>::distance_point_to_point ( p, f, SnakeSpace<17>::ScanDirection::ne ) << nl;
    std::cout << SnakeSpace<17>::distance_point_to_point ( p, f, SnakeSpace<17>::ScanDirection::ea ) << nl;
    std::cout << SnakeSpace<17>::distance_point_to_point ( p, f, SnakeSpace<17>::ScanDirection::nw ) << nl;
    std::cout << SnakeSpace<17>::distance_point_to_point ( p, f, SnakeSpace<17>::ScanDirection::so ) << nl;
    std::cout << SnakeSpace<17>::distance_point_to_point ( p, f, SnakeSpace<17>::ScanDirection::sw ) << nl;
    std::cout << SnakeSpace<17>::distance_point_to_point ( p, f, SnakeSpace<17>::ScanDirection::we ) << nl;
    std::cout << SnakeSpace<17>::distance_point_to_point ( p, f, SnakeSpace<17>::ScanDirection::se ) << nl;

    float dists[ 8 ];

    SnakeSpace<17>::distances_point_to_point ( dists, p, f );

    for ( auto f : dists )
        std::cout << f << ' ';
    std::cout << nl;

    ss.print ( );

    std::cout << ss.distance_to_body ( SnakeSpace<17>::ScanDirection::no ) << nl;
    std::cout << ss.distance_to_body ( SnakeSpace<17>::ScanDirection::ne ) << nl;
    std::cout << ss.distance_to_body ( SnakeSpace<17>::ScanDirection::ea ) << nl;
    std::cout << ss.distance_to_body ( SnakeSpace<17>::ScanDirection::nw ) << nl;
    std::cout << ss.distance_to_body ( SnakeSpace<17>::ScanDirection::so ) << nl;
    std::cout << ss.distance_to_body ( SnakeSpace<17>::ScanDirection::sw ) << nl;
    std::cout << ss.distance_to_body ( SnakeSpace<17>::ScanDirection::we ) << nl;
    std::cout << ss.distance_to_body ( SnakeSpace<17>::ScanDirection::se ) << nl;

    float dists2[ 8 ];

    ss.distances_to_body ( dists2 );

    for ( auto f : dists2 )
        std::cout << f << ' ';
    std::cout << nl;

    ss.print ( );

    // ss.run ( );


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
    */
    return EXIT_SUCCESS;
}

/*


template<int NumInput, int NumNeurons, int NumOutput>
struct FullyConnectedNeuralNetwork {

    static_assert ( NumNeurons >= NumOutput, "number of neurons needs to be equal or larger than the number of required
outputs" );

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
