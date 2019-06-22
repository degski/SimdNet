
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
#include <cstring>

#include <algorithm>
#include <sax/iostream.hpp>
#include <string>
#include <type_traits>

#include "ring_span.hpp"

// Make ring_span from std arrays.
template<typename Popper, typename T, size_t N>
[[nodiscard]] constexpr nonstd::ring_span<T, Popper> make_ring_span ( std::array<T, N> & std_arr_ ) noexcept {
    auto arr = std_arr_.data ( );
    return { arr, arr + N, arr, N };
}

// A thread-safe singleton ring(-span).
template<typename Popper, typename T, size_t N>
[[nodiscard]] nonstd::ring_span<T, Popper> make_ring_span ( ) noexcept {
    static thread_local std::array<T, N> & std_arr{};
    return { std_arr.data ( ), std_arr.data ( ) + N, std_arr.data ( ), N };
}

#include <plf/plf_nanotimer.h>

#include <sax/uniform_int_distribution.hpp>

#include "fcc.hpp"
#include "globals.hpp"
#include "rng.hpp"

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
    auto idx = [] ( ) noexcept { return static_cast<char> ( sax::uniform_int_distribution<int>{ -B, B }( Rng::gen ( ) ) ); };
    return { idx ( ), idx ( ) };
}

template<int FieldSize, int NumInput, int NumNeurons, int NumOutput>
struct SnakeSpace {

    static_assert ( FieldSize % 2 != 0, "uneven size only" );

    static constexpr int FieldRadius = FieldSize / 2;

    enum class MoveDirection : int { no, ea, so, we };

    using SnakeBody = nonstd::ring_span<Point, nonstd::null_popper<Point>>;

    using pointer         = float *;
    using const_pointer   = float const *;
    using reference       = float &;
    using const_reference = float const &;

    using TheBrain = FullyConnectedNeuralNetwork<NumInput, NumNeurons, NumOutput>;
    using WorkArea = InputBiasOutput<NumInput, NumNeurons, NumOutput>;

    SnakeSpace ( ) noexcept : m_snake_body{ make_ring_span<nonstd::null_popper<Point>> ( m_snake_body_data ) } {}

    [[nodiscard]] inline bool in_range ( Point const & p_ ) const noexcept {
        return p_.x >= -FieldRadius and p_.y >= -FieldRadius and p_.x <= FieldRadius and p_.y <= FieldRadius;
    }

    [[nodiscard]] inline bool snake_body_contains ( Point const & p_ ) const noexcept {
        return std::find ( std::cbegin ( m_snake_body ), std::cend ( m_snake_body ), p_ ) != std::cend ( m_snake_body );
    }

    // Returns whether the head is at the same position as any of the body parts.
    [[nodiscard]] inline bool snake_body_not_crossing ( ) const noexcept {
        return std::find ( std::cbegin ( m_snake_body ) + 1, std::cend ( m_snake_body ), m_snake_body.front ( ) ) ==
               std::cend ( m_snake_body );
    }

    [[nodiscard]] inline bool valid_empty_point ( Point const & p_ ) const noexcept {
        return in_range ( p_ ) and not snake_body_contains ( p_ );
    }

    void random_food ( ) noexcept {
        Point f = random_point<FieldRadius> ( );
        while ( snake_body_contains ( f ) )
            f = random_point<FieldRadius> ( );
        m_food = f;
    }

    void init_run ( ) noexcept {
        m_move_count = 0;
        m_energy     = 100;
        m_direction  = static_cast<MoveDirection> ( sax::uniform_int_distribution<int>{ 0, 3 }( Rng::gen ( ) ) );
        m_snake_body = SnakeBody{ std::begin ( m_snake_body_data ), std::end ( m_snake_body_data ) };
        m_snake_body.emplace_front ( random_point<FieldRadius - 6> ( ) ); // the new tail.
        m_snake_body.emplace_front ( extend_head ( ) );
        m_snake_body.emplace_front ( extend_head ( ) ); // the new head.
        random_food ( );
    }

    [[nodiscard]] inline Point extend_head ( ) const noexcept {
        switch ( m_direction ) {
            case MoveDirection::no: return Point{ +0, +1 } + m_snake_body.front ( );
            case MoveDirection::ea: return Point{ +1, +0 } + m_snake_body.front ( );
            case MoveDirection::so: return Point{ +0, -1 } + m_snake_body.front ( );
            case MoveDirection::we: return Point{ -1, +0 } + m_snake_body.front ( );
        }
        return { 0, 0 };
    }

    [[nodiscard]] inline bool is_not_dead ( ) const noexcept {
        return m_energy and in_range ( m_snake_body.front ( ) ) and snake_body_not_crossing ( );
    }

    // Returns the dead(false)/alive(true) status.
    bool move ( ) noexcept {
        ++m_move_count;
        --m_energy;
        m_snake_body.emplace_front ( extend_head ( ) );
        if ( is_not_dead ( ) ) {
            if ( m_snake_body.front ( ) != m_food ) {
                m_snake_body.pop_back ( );
                return true;
            }
            else {
                m_energy += EnergyTopUp;
                random_food ( );
                return true;
            }
        }
        return false;
    }

    struct Changes {
        Point old_head, new_head;
        bool has_eaten;
        Point old_tail;
    };

    // Returns the dead(false)/alive(true) status.
    bool move_display ( ) noexcept {
        ++m_move_count;
        --m_energy;
        m_changes.old_head = m_snake_body.front ( );
        m_snake_body.emplace_front ( extend_head ( ) );
        m_changes.new_head = m_snake_body.front ( );
        if ( is_not_dead ( ) ) {
            if ( m_snake_body.front ( ) != m_food ) {
                m_changes.has_eaten = false;
                m_changes.old_tail  = m_snake_body.back ( );
                m_snake_body.pop_back ( );
                return true;
            }
            else {
                m_changes.has_eaten = true;
                m_energy += EnergyTopUp;
                random_food ( );
                return true;
            }
        }
        return false;
    }

    // left = 0, ahead = 1, right = 2
    [[nodiscard]] inline MoveDirection decide_direction_3 ( const_pointer o_ ) const noexcept {
        switch ( m_direction ) {
            case MoveDirection::no:
                return o_[ 0 ] > o_[ 1 ] ? ( o_[ 0 ] > o_[ 2 ] ? MoveDirection::ea : MoveDirection::we )
                                         : ( o_[ 1 ] > o_[ 2 ] ? MoveDirection::no : MoveDirection::we );
            case MoveDirection::ea:
                return o_[ 0 ] > o_[ 1 ] ? ( o_[ 0 ] > o_[ 2 ] ? MoveDirection::no : MoveDirection::so )
                                         : ( o_[ 1 ] > o_[ 2 ] ? MoveDirection::ea : MoveDirection::so );
            case MoveDirection::so:
                return o_[ 0 ] > o_[ 1 ] ? ( o_[ 0 ] > o_[ 2 ] ? MoveDirection::we : MoveDirection::ea )
                                         : ( o_[ 1 ] > o_[ 2 ] ? MoveDirection::so : MoveDirection::ea );
            case MoveDirection::we:
                return o_[ 0 ] > o_[ 1 ] ? ( o_[ 0 ] > o_[ 2 ] ? MoveDirection::so : MoveDirection::no )
                                         : ( o_[ 1 ] > o_[ 2 ] ? MoveDirection::we : MoveDirection::no );
        }
        return MoveDirection::no;
    }

    [[nodiscard]] inline MoveDirection decide_direction_4 ( const_pointer o_ ) const noexcept {
        return o_[ 1 ] > o_[ 0 ] ? ( o_[ 3 ] > o_[ 2 ] ? ( o_[ 3 ] > o_[ 1 ] ? MoveDirection::we : MoveDirection::ea )
                                                       : ( o_[ 2 ] > o_[ 1 ] ? MoveDirection::so : MoveDirection::ea ) )
                                 : ( o_[ 3 ] > o_[ 2 ] ? ( o_[ 3 ] > o_[ 0 ] ? MoveDirection::we : MoveDirection::no )
                                                       : ( o_[ 2 ] > o_[ 0 ] ? MoveDirection::so : MoveDirection::no ) );
    }

    [[nodiscard]] inline MoveDirection decide_direction ( const_pointer o_ ) const noexcept {
        if constexpr ( 3 == NumOutput ) {
            return decide_direction_3 ( o_ );
        }
        else {
            return decide_direction_4 ( o_ );
        }
    }

    // Return the fitness of the network.
    [[nodiscard]] float run ( TheBrain * const brain_ ) noexcept {
        static thread_local WorkArea work_area;
        constexpr int s = 3;
        int r           = 0;
        for ( int i = 0; i < s; ++i ) {
            init_run ( );
            while ( move ( ) ) {                        // As long as not dead.
                gather_input_10 ( work_area.data ( ) ); // Observe the environment.
                m_direction =
                    decide_direction ( brain_->feed_forward ( work_area.data ( ) ) ); // Run the data and decide where to go,
            }                                                                           // and change direction.
            r += m_snake_body.size ( );
        }
        return static_cast<float> ( r ) / static_cast<float> ( s );
    }

    void run_display ( TheBrain * const brain_ ) noexcept {
        static thread_local WorkArea work_area;
        init_run ( );
        set_cursor_position ( 0, 0 );
        print ( );
        while ( move_display ( ) ) {                // As long as not dead.
            gather_input_10 ( work_area.data ( ) ); // Observe the environment.
            m_direction = decide_direction (
                brain_->feed_forward ( work_area.data ( ) ) ); // Run the data and decide where to go, and change direction.
            print_update ( );
            sleep_for_milliseconds ( 25 );
        }
    }

    private:
    // Manhattan distance (activation) between points.
    [[nodiscard]] static std::tuple<int, float> distance_point_to_point_8 ( Point const & p0_, Point const & p1_ ) noexcept {
        Point const s = p0_ - p1_;
        if ( 0 == s.x )
            return s.y < 0 ? std::tuple<int, float> ( 0, 1.0f / -s.y ) : std::tuple<int, float> ( 4, 1.0f / +s.y );
        if ( s.x == s.y )
            return s.y < 0 ? std::tuple<int, float> ( 1, 0.5f / -s.y ) : std::tuple<int, float> ( 5, 0.5f / +s.y );
        if ( 0 == s.y )
            return s.x < 0 ? std::tuple<int, float> ( 2, 1.0f / -s.x ) : std::tuple<int, float> ( 6, 1.0f / +s.x );
        if ( s.x == -s.y )
            return s.x < 0 ? std::tuple<int, float> ( 3, 0.5f / -s.x ) : std::tuple<int, float> ( 7, 0.5f / +s.x );
        return std::tuple<int, float> ( 0, 0.0f ); // Default is north, but set to zero, which avoids any checking.
    }

    // Input (activation) for distances to wall.
    void distances_to_wall_8 ( pointer data_ ) const noexcept {
        Point const & head = m_snake_body.front ( );
        data_[ 0 ]         = 1.0f / ( FieldRadius - head.y + 1 );
        data_[ 1 ]         = 1.0f / ( 2 * std::min ( FieldRadius - head.x, FieldRadius - head.y ) + 1 );
        data_[ 2 ]         = 1.0f / ( FieldRadius - head.x + 1 );
        data_[ 3 ]         = 1.0f / ( 2 * std::min ( FieldRadius - head.x, FieldRadius + head.y ) + 1 );
        data_[ 4 ]         = 1.0f / ( FieldRadius + head.y + 1 );
        data_[ 5 ]         = 1.0f / ( 2 * std::min ( FieldRadius + head.x, FieldRadius + head.y ) + 1 );
        data_[ 6 ]         = 1.0f / ( FieldRadius + head.x + 1 );
        data_[ 7 ]         = 1.0f / ( 2 * std::min ( FieldRadius + head.x, FieldRadius - head.y ) + 1 );
    }

    // Input (activation) for distances to food.
    void distances_to_food_8 ( pointer data_ ) const noexcept {
        auto const [ dir, val ] = distance_point_to_point_8 ( m_snake_body.front ( ), m_food );
        data_[ dir ]            = val;
    }

    // Input (activation) for distances to body.
    void distances_to_body_8 ( pointer data_ ) const noexcept {
        Point const & head = m_snake_body.front ( );
        auto const end     = std::end ( m_snake_body );
        auto it            = std::begin ( m_snake_body ); // This assumes the length of the snake is at least 2.
        while ( ++it != end ) {
            auto const [ dir, val ] = distance_point_to_point_8 ( head, *it );
            if ( val > data_[ dir ] )
                data_[ dir ] = val;
        }
    }

    // Encodes, where the food is in relation to the direction the snake is
    // moving in. So, 'in front', 'to the left', 'to the right' and 'behind'.
    void gather_input_10 ( pointer data_ ) const noexcept {
        Point const f = m_snake_body.front ( ), d = m_food - f;
        switch ( m_direction ) {
            case MoveDirection::no:
                data_[ 0 ] = static_cast<float> ( valid_empty_point ( Point{ static_cast<char> ( f.x - 1 ), f.y } ) );
                data_[ 1 ] = static_cast<float> ( valid_empty_point ( Point{ f.x, static_cast<char> ( f.y + 1 ) } ) );
                data_[ 2 ] = static_cast<float> ( valid_empty_point ( Point{ static_cast<char> ( f.x + 1 ), f.y } ) );
                data_[ 3 ] = static_cast<float> ( static_cast<int> ( d.y > 0 ) * 2 - 1 ); // no
                data_[ 4 ] = static_cast<float> ( static_cast<int> ( d.x > 0 ) * 2 - 1 ); // ea
                data_[ 5 ] = static_cast<float> ( static_cast<int> ( d.y < 0 ) * 2 - 1 ); // so
                data_[ 6 ] = static_cast<float> ( static_cast<int> ( d.x < 0 ) * 2 - 1 ); // we
                data_[ 7 ] = +1.0f;
                data_[ 8 ] = +0.0f;
                data_[ 9 ] = 1.0f / ( 1.0f + m_energy );
                return;
            case MoveDirection::ea:
                data_[ 0 ] = static_cast<float> ( valid_empty_point ( Point{ f.x, static_cast<char> ( f.y + 1 ) } ) );
                data_[ 1 ] = static_cast<float> ( valid_empty_point ( Point{ static_cast<char> ( f.x + 1 ), f.y } ) );
                data_[ 2 ] = static_cast<float> ( valid_empty_point ( Point{ f.x, static_cast<char> ( f.y - 1 ) } ) );
                data_[ 3 ] = static_cast<float> ( static_cast<int> ( d.x > 0 ) * 2 - 1 ); // ea
                data_[ 4 ] = static_cast<float> ( static_cast<int> ( d.y < 0 ) * 2 - 1 ); // so
                data_[ 5 ] = static_cast<float> ( static_cast<int> ( d.x < 0 ) * 2 - 1 ); // we
                data_[ 6 ] = static_cast<float> ( static_cast<int> ( d.y > 0 ) * 2 - 1 ); // no
                data_[ 7 ] = +0.0f;
                data_[ 8 ] = +1.0f;
                data_[ 9 ] = 1.0f / ( 1.0f + m_energy );
                return;
            case MoveDirection::so:
                data_[ 0 ] = static_cast<float> ( valid_empty_point ( Point{ static_cast<char> ( f.x + 1 ), f.y } ) );
                data_[ 1 ] = static_cast<float> ( valid_empty_point ( Point{ f.x, static_cast<char> ( f.y - 1 ) } ) );
                data_[ 2 ] = static_cast<float> ( valid_empty_point ( Point{ static_cast<char> ( f.x - 1 ), f.y } ) );
                data_[ 3 ] = static_cast<float> ( static_cast<int> ( d.y < 0 ) * 2 - 1 ); // so
                data_[ 4 ] = static_cast<float> ( static_cast<int> ( d.x < 0 ) * 2 - 1 ); // we
                data_[ 5 ] = static_cast<float> ( static_cast<int> ( d.y > 0 ) * 2 - 1 ); // no
                data_[ 6 ] = static_cast<float> ( static_cast<int> ( d.x > 0 ) * 2 - 1 ); // ea
                data_[ 7 ] = -1.0f;
                data_[ 8 ] = +0.0f;
                data_[ 9 ] = 1.0f / ( 1.0f + m_energy );
                return;
            case MoveDirection::we:
                data_[ 0 ] = static_cast<float> ( valid_empty_point ( Point{ f.x, static_cast<char> ( f.y - 1 ) } ) );
                data_[ 1 ] = static_cast<float> ( valid_empty_point ( Point{ static_cast<char> ( f.x - 1 ), f.y } ) );
                data_[ 2 ] = static_cast<float> ( valid_empty_point ( Point{ f.x, static_cast<char> ( f.y + 1 ) } ) );
                data_[ 3 ] = static_cast<float> ( static_cast<int> ( d.x < 0 ) * 2 - 1 ); // we
                data_[ 4 ] = static_cast<float> ( static_cast<int> ( d.y > 0 ) * 2 - 1 ); // no
                data_[ 5 ] = static_cast<float> ( static_cast<int> ( d.x > 0 ) * 2 - 1 ); // ea
                data_[ 6 ] = static_cast<float> ( static_cast<int> ( d.y < 0 ) * 2 - 1 ); // so
                data_[ 7 ] = +0.0f;
                data_[ 8 ] = -1.0f;
                data_[ 9 ] = 1.0f / ( 1.0f + m_energy );
                return;
        }
    }

    void encode_current_direction_2 ( pointer data_ ) const noexcept {
        switch ( m_direction ) {
            case MoveDirection::no:
                data_[ 0 ] = +1.0f;
                data_[ 1 ] = +0.0f;
                return;
            case MoveDirection::ea:
                data_[ 0 ] = +0.0f;
                data_[ 1 ] = +1.0f;
                return;
            case MoveDirection::so:
                data_[ 0 ] = -1.0f;
                data_[ 1 ] = +0.0f;
                return;
            case MoveDirection::we:
                data_[ 0 ] = +0.0f;
                data_[ 1 ] = -1.0f;
                return;
        }
    }

    void encode_energy_1 ( pointer data_ ) const noexcept { data_[ 0 ] = 1.0f / ( 1.0f + m_energy ); }

    public:
    void gather_input ( pointer d_ ) const noexcept {
        distances_to_wall_8 ( d_ );
        std::memset ( d_ + 8, 0, 16 * sizeof ( float ) );
        distances_to_food_8 ( d_ + 8 );
        distances_to_body_8 ( d_ + 16 );
        encode_current_direction_2 ( d_ + 24 );
        encode_energy_1 ( d_ + 26 );
    }

    void print ( ) const noexcept {
        static bool _ = hide_cursor ( ); // Call only once.
        for ( int y = -FieldRadius; y <= FieldRadius; ++y ) {
            for ( int x = -FieldRadius; x <= FieldRadius; ++x ) {
                Point const p{ static_cast<char> ( x ), static_cast<char> ( y ) };
                if ( p == m_food )
                    std::wprintf ( L" \u25B2 " );
                else if ( snake_body_contains ( p ) )
                    // if ( p == m_snake_body.front ( ) )
                    std::wprintf ( L" \u25A0 " );
                // else
                // std::wprintf ( L" \u25A1 " );
                else
                    std::wprintf ( L" \u00B7 " );
            }
            std::wprintf ( L"\n" );
        }
        std::wprintf ( L"\n" );
    }

    void print_update ( ) const noexcept {
        set_cursor_position ( ( m_changes.new_head.x + FieldRadius ) * 3 + 1, m_changes.new_head.y + FieldRadius );
        std::putwchar ( L'\u25A0' );
        // set_cursor_position ( ( m_changes.old_head.x + FieldRadius ) * 3 + 1, m_changes.old_head.y + FieldRadius );
        // std::putwchar ( L'\u25A1' );
        if ( m_changes.has_eaten ) {
            set_cursor_position ( ( m_food.x + FieldRadius ) * 3 + 1, m_food.y + FieldRadius );
            std::putwchar ( L'\u25B2' );
        }
        else {
            set_cursor_position ( ( m_changes.old_tail.x + FieldRadius ) * 3 + 1, m_changes.old_tail.y + FieldRadius );
            std::putwchar ( L'\u00B7' );
        }
        set_cursor_position ( 1, FieldSize + 2 );
    }

    static constexpr int EnergyTopUp = 100;

    int m_move_count, m_energy;
    MoveDirection m_direction;
    std::array<Point, 384> m_snake_body_data;
    SnakeBody m_snake_body;
    Point m_food;
    Changes m_changes;
};
