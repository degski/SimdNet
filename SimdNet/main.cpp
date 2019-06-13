
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

template<int N>
int dist ( float v_ ) noexcept {
    constexpr float n     = N;
    constexpr float sum_n = 0.5f * n * ( n + 1.0f );
    constexpr float a     = ( sum_n - n * n ) / ( ( n * n ) / sum_n - n );
    constexpr float b     = n - a;
    return a * v_ * v_ + b * v_;
}

int main5658 ( ) {

    Population<4'096 * 8, 39, 24, 5, 4> p;

    p.run ( );

    return EXIT_SUCCESS;
}

// https : // oroboro.com/non-uniform-random-numbers/

#define U32_MAX 0xFFFFFFFF

// Since we are using fixed point math, we first implement
// a fixed 0.32 point inversion:  1/(n-1)
#define DIST_INV( n ) ( 0xFFFFFFFF / ( ( n ) -1 ) )

class Distribution {
    public:
    Distribution ( ) { ; }
    Distribution ( std::uint32_t a, std::uint32_t b, std::uint32_t p ) {
        mA = a;
        mB = b, mProb = p;
    }

    std::uint32_t mA;
    std::uint32_t mB;
    std::uint32_t mProb;
};

static void computeBiDist ( std::vector<std::uint32_t> & p, std::uint32_t n, Distribution & dist, std::uint32_t aIdx,
                            std::uint32_t bIdx ) {
    dist.mA = aIdx;
    dist.mB = bIdx;
    if ( aIdx == bIdx ) {
        dist.mProb = 0;
    }
    else {
        if ( ( ( p[ aIdx ] >> 1 ) * ( n - 1 ) ) >= 0x80000000 )
            dist.mProb = 0xFFFFFFFF;
        else
            dist.mProb = p[ aIdx ] * ( n - 1 );
        p[ bIdx ] -= ( DIST_INV ( n ) - p[ aIdx ] );
    }
    p[ aIdx ] = 0;
}

static void normProbs ( std::vector<std::uint32_t> & probs ) {
    std::uint32_t scale  = 0;
    std::uint32_t err    = 0;
    std::uint32_t shift  = 0;
    std::uint32_t max    = 0;
    std::uint32_t maxIdx = 0;

    // how many non-zero probabilities?
    std::uint32_t numNonZero = 0;
    for ( std::uint32_t i = 0; i < probs.size ( ); i++ )
        if ( probs[ i ] )
            numNonZero++;

    if ( numNonZero == 0 ) {
        // degenerate all zero probability array.
        // Can't do anything with it... result is undefined
        assert ( 0 );
        return;
    }

    if ( numNonZero == 1 ) {
        // trivial case with only one real prob - handle special because
        // computation would overflow below anyway.
        for ( std::uint32_t i = 0; i < probs.size ( ); i++ )
            probs[ i ] = probs[ i ] ? U32_MAX : 0;
        return;
    }

// figure out the scale
again:
    for ( std::uint32_t i = 0; i < probs.size ( ); i++ )
        scale += ( ( probs[ i ] << shift ) >> 8 );

    if ( ( scale < 0xFFFF ) && ( shift < 24 ) ) {
        shift += 8;
        goto again;
    }

    assert ( scale );

    scale = 0x10000000 / ( ( scale + 0x7FF ) >> 12 );

    // apply it
    for ( std::uint32_t i = 0; i < probs.size ( ); i++ ) {
        probs[ i ] = ( ( ( probs[ i ] << shift ) + 0x7FFF ) >> 16 ) * scale;
        err += probs[ i ];
        if ( probs[ i ] > max ) {
            max    = probs[ i ];
            maxIdx = i;
        }
    }

    // correct any accumulated error - it should be negligible. Add it
    // to the largest probability where it will be least noticed.
    probs[ maxIdx ] -= err;
}

class KxuRand {
    public:
    virtual ~KxuRand ( ) { ; }
    virtual std::uint32_t getRandom ( ) = 0;
    virtual double getRandomUnit ( ) { return ( ( ( double ) getRandom ( ) ) / U32_MAX ); }
};

class KxuRandUniform : public KxuRand {
    public:
    virtual ~KxuRandUniform ( ) { ; }
    virtual void setSeed ( std::uint32_t seed ) = 0;

    std::uint32_t getRandomInRange ( std::uint32_t n ) {
        std::uint64_t v = getRandom ( );
        v *= n;
        return std::uint32_t ( v >> 32 );
    }
    std::uint32_t getRandomInRange ( std::uint32_t start, std::uint32_t end ) { return getRandomInRange ( end - start ) + start; }
};

// a dead simple Linear Congruent random number generator
class KxuLCRand : public KxuRandUniform {
    public:
    KxuLCRand ( std::uint32_t seed = 555 ) { setSeed ( seed ); }
    void setSeed ( std::uint32_t seed ) {
        if ( !seed )
            seed = 0x333;
        mState = seed | 1;
    }
    std::uint32_t getRandom ( ) {
        mState = ( mState * 69069 ) + 1;
        return mState;
    }

    private:
    std::uint32_t mState;
};

double fixedToFloat_0_32 ( std::uint32_t val ) { return ( ( ( double ) val ) / U32_MAX ); }
std::uint32_t floatToFixed_0_32 ( double val ) { return ( std::uint32_t ) ( val * U32_MAX ); }

class KxuNuRand : public KxuRand {
    public:
    KxuNuRand ( const std::vector<std::uint32_t> & dist, KxuRandUniform * rand );
    std::uint32_t getRandom ( );

    protected:
    std::vector<Distribution> mDist;
    KxuRandUniform * mRand;
};

std::uint32_t KxuNuRand::getRandom ( ) {
    const Distribution & dist = mDist[ mRand->getRandomInRange ( static_cast<std::uint32_t> ( mDist.size ( ) ) ) ];
    return ( mRand->getRandom ( ) <= dist.mProb ) ? dist.mA : dist.mB;
}

KxuNuRand::KxuNuRand ( const std::vector<std::uint32_t> & dist, KxuRandUniform * rand ) {
    mRand = rand;

    if ( dist.size ( ) == 1 ) {
        // handle a the special case of just one symbol.
        mDist.emplace_back ( Distribution ( 0, 0, 0 ) );
    }
    else {

        // The non-uniform distribution is passed in as an argument to the
        // constructor. This is a series of integers in the desired proportions.
        // Normalize these into a series of probabilities that sum to 1, expressed
        // in 0.32 fixed point.
        std::vector<std::uint32_t> p;
        p = dist;
        normProbs ( p );

        // Then we count up the number of non-zero probabilities so that we can
        // figure out the number of distributions we need.
        std::uint32_t numDistros = 0;
        for ( std::uint32_t i = 0; i < p.size ( ); i++ )
            if ( p[ i ] )
                numDistros++;
        if ( numDistros < 2 )
            numDistros = 2;
        std::uint32_t thresh = DIST_INV ( numDistros );

        // reserve space for the distributions.
        mDist.resize ( numDistros - 1 );

        std::uint32_t aIdx = 0;
        std::uint32_t bIdx = 0;
        for ( std::uint32_t i = 0; i < mDist.size ( ); i++ ) {
            // find a small prob, non-zero preferred
            while ( aIdx < p.size ( ) - 1 ) {
                if ( ( p[ aIdx ] < thresh ) && p[ aIdx ] )
                    break;
                aIdx++;
            }
            if ( p[ aIdx ] >= thresh ) {
                aIdx = 0;
                while ( aIdx < p.size ( ) - 1 ) {
                    if ( p[ aIdx ] < thresh )
                        break;
                    aIdx++;
                }
            }

            // find a prob that is not aIdx, and the sum is more than thresh.
            while ( bIdx < p.size ( ) - 1 ) {
                if ( bIdx == aIdx ) {
                    bIdx++;
                    continue;
                } // can't be aIdx
                if ( ( ( p[ aIdx ] >> 1 ) + ( p[ bIdx ] >> 1 ) ) >= ( thresh >> 1 ) )
                    break;
                bIdx++; // find a sum big enough or equal
            }

            // We've selected 2 symbols, at indexes aIdx, and bIdx.
            // This function will initialize a new binary distribution, and make
            // the appropriate adjustments to the input non-uniform distribution.
            computeBiDist ( p, numDistros, mDist[ i ], aIdx, bIdx );

            if ( ( bIdx < aIdx ) && ( p[ bIdx ] < thresh ) )
                aIdx = bIdx;
            else
                aIdx++;
        }
    }
}

#if 0

 /******************************************************************************
 * File: AliasMethod.java
 * Author: Keith Schwarz (htiek@cs.stanford.edu)
 *
 * An implementation of the alias method implemented using Vose's algorithm.
 * The alias method allows for efficient sampling of random values from a
 * discrete probability distribution (i.e. rolling a loaded die) in O(1) time
 * each after O(n) preprocessing time.
 *
 * For a complete writeup on the alias method, including the intuition and
 * important proofs, please see the article "Darts, Dice, and Coins: Sampling
 * from a Discrete Distribution" at
 *
 *                 http://www.keithschwarz.com/darts-dice-coins/
 */
import java.util.*;

public
final class AliasMethod {
    /* The random number generator used to sample from the distribution. */
    private
    final Random random;

    /* The probability and alias tables. */
    private
    final int[] alias;
    private
    final double[] probability;

    /**
     * Constructs a new AliasMethod to sample from a discrete distribution and
     * hand back outcomes based on the probability distribution.
     * <p>
     * Given as input a list of probabilities corresponding to outcomes 0, 1,
     * ..., n - 1, this constructor creates the probability and alias tables
     * needed to efficiently sample from this distribution.
     *
     * @param probabilities The list of probabilities.
     */
    public
    AliasMethod ( List<Double> probabilities ) { this( probabilities, new Random ( ) ); }

    /**
     * Constructs a new AliasMethod to sample from a discrete distribution and
     * hand back outcomes based on the probability distribution.
     * <p>
     * Given as input a list of probabilities corresponding to outcomes 0, 1,
     * ..., n - 1, along with the random number generator that should be used
     * as the underlying generator, this constructor creates the probability
     * and alias tables needed to efficiently sample from this distribution.
     *
     * @param probabilities The list of probabilities.
     * @param random The random number generator
     */
    public
    AliasMethod ( List<Double> probabilities, Random random ) {
        /* Begin by doing basic structural checks on the inputs. */
        if ( probabilities == null || random == null )
            throw new NullPointerException ( );
        if ( probabilities.size ( ) == 0 )
            throw new IllegalArgumentException ( "Probability vector must be nonempty." );

        /* Allocate space for the probability and alias tables. */
        probability = new double[ probabilities.size ( ) ];
        alias       = new int[ probabilities.size ( ) ];

        /* Store the underlying generator. */
        this.random = random;

        /* Compute the average probability and cache it for later use. */
        final double average = 1.0 / probabilities.size ( );

        /* Make a copy of the probabilities list, since we will be making
         * changes to it.
         */
        probabilities = new ArrayList<Double> ( probabilities );

        /* Create two stacks to act as worklists as we populate the tables. */
        Deque<Integer> small = new ArrayDeque<Integer> ( );
        Deque<Integer> large = new ArrayDeque<Integer> ( );

        /* Populate the stacks with the input probabilities. */
        for ( int i = 0; i < probabilities.size ( ); ++i ) {
            /* If the probability is below the average probability, then we add
             * it to the small list; otherwise we add it to the large list.
             */
            if ( probabilities.get ( i ) >= average )
                large.add ( i );
            else
                small.add ( i );
        }

        /* As a note: in the mathematical specification of the algorithm, we
         * will always exhaust the small list before the big list.  However,
         * due to floating point inaccuracies, this is not necessarily true.
         * Consequently, this inner loop (which tries to pair small and large
         * elements) will have to check that both lists aren't empty.
         */
        while ( !small.isEmpty ( ) && !large.isEmpty ( ) ) {
            /* Get the index of the small and the large probabilities. */
            int less = small.removeLast ( );
            int more = large.removeLast ( );

            /* These probabilities have not yet been scaled up to be such that
             * 1/n is given weight 1.0.  We do this here instead.
             */
            probability[ less ] = probabilities.get ( less ) * probabilities.size ( );
            alias[ less ]       = more;

            /* Decrease the probability of the larger one by the appropriate
             * amount.
             */
            probabilities.set ( more, ( probabilities.get ( more ) + probabilities.get ( less ) ) - average );

            /* If the new probability is less than the average, add it into the
             * small list; otherwise add it to the large list.
             */
            if ( probabilities.get ( more ) >= 1.0 / probabilities.size ( ) )
                large.add ( more );
            else
                small.add ( more );
        }

        /* At this point, everything is in one list, which means that the
         * remaining probabilities should all be 1/n.  Based on this, set them
         * appropriately.  Due to numerical issues, we can't be sure which
         * stack will hold the entries, so we empty both.
         */
        while ( !small.isEmpty ( ) )
            probability[ small.removeLast ( ) ] = 1.0;
        while ( !large.isEmpty ( ) )
            probability[ large.removeLast ( ) ] = 1.0;
    }

    /**
     * Samples a value from the underlying distribution.
     *
     * @return A random value sampled from the underlying distribution.
     */
    public
    int next ( ) {
        /* Generate a fair die roll to determine which column to inspect. */
        int column = random.nextInt ( probability.length );

        /* Generate a biased coin toss to determine which option to pick. */
        boolean coinToss = random.nextDouble ( ) < probability[ column ];

        /* Based on the outcome, return either the column or its alias. */
        return coinToss ? column : alias[ column ];
    }
}

#endif

// http://www.keithschwarz.com/darts-dice-coins/

// The probability and alias tables.
template<typename T = int, typename U = float>
struct VoseTables {
    std::vector<U> m_probability{};
    std::vector<T> m_alias{};
    explicit VoseTables ( T const n_ ) :
        m_probability ( static_cast<std::size_t> ( n_ + 1 ), U{ 0 } ), m_alias ( static_cast<std::size_t> ( n_ + 1 ), T{ 0 } ) {
        m_probability.resize ( static_cast<std::size_t> ( n_ ) );
        m_alias.resize ( static_cast<std::size_t> ( n_ ) );
    }

    [[nodiscard]] int size ( ) const noexcept { return static_cast<int> ( m_probability.size ( ) ); }
};

template<typename T = int, typename U = float>
VoseTables<T, U> init ( std::vector<U> const & pset_ ) noexcept {

    assert ( pset_.size ( ) > 0u );

    std::vector<U> pset{ pset_ };

    T const n = static_cast<T> ( pset.size ( ) );

    std::for_each ( std::execution::par_unseq, std::begin ( pset ), std::end ( pset ), [f = static_cast<float>( n )]( U & v ) { return v *= f; } );

    std::vector<int> lrg, sml;

    lrg.reserve ( n );
    sml.reserve ( n );

    T i = 0;

    for ( auto const p : pset ) {
        if ( p >= U{ 1 } )
            lrg.push_back ( i );
        else
            sml.push_back ( i );
        ++i;
    }

    VoseTables<T, U> tables ( n );

    while ( lrg.size ( ) and sml.size ( ) ) {

        T const l = sml.back ( );
        sml.pop_back ( );
        T const g = lrg.back ( );
        lrg.pop_back ( );

        tables.m_probability[ l ] = pset [ l ];
        tables.m_alias[ l ] = g;

        pset[ g ] = ( pset[ g ] + pset[ l ] ) - U{ 1 };

        if ( pset[ g ] >= U{ 1 } )
            lrg.push_back ( g );
        else
            sml.push_back ( g );
    }

    while ( lrg.size ( ) ) {
        tables.m_probability[ lrg.back ( ) ] = U{ 1 };
        lrg.pop_back ( );
    }
    while ( sml.size ( ) ) {
        tables.m_probability[ sml.back ( ) ] = U{ 1 };
        sml.pop_back ( );
    }

    tables.m_probability.push_back ( tables.m_probability.back ( ) );
    tables.m_alias.push_back ( tables.m_alias.back ( ) );

    return tables;
}

template<typename T = int, typename U = float>
int next ( VoseTables<T, U> & dis_ ) {
    int const column = sax::uniform_int_distribution<int> ( 0, dis_.size ( ) - 1 ) ( Rng::gen ( ) );
    return Rng::bernoulli ( dis_.m_probability[ column ] ) ? column : dis_.m_alias[ column ];
}

int main ( ) {

    auto dis = init ( std::vector<float>{ 10.0f, 20.0f, 30.0f } );

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
