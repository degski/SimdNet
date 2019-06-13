#include "gp.h"



typedef struct {

    char * id;
    __int32 * set;

    float * prob;
    __int32 * alias;

    float n;

    UT_hash_handle hh;

} vose_t;


vose_t *vose_pool = NULL;


vose_t *initiate_vose ( char *id, const unsigned int n, const int *vset, const float *pset ) {

    unsigned int i, j, l = 0, s = 0;
    float r = 0.0f;
    vose_t *vs;
    unsigned int len = n << 2;

    vs = ( vose_t * ) malloc ( sizeof ( vose_t ) );

    vs->n = ( float ) n;

    vs->set = ( int * ) malloc ( len );
    memcpy ( vs->set, vset, len );

    float p [ len ];
    memcpy ( p, pset, len );

    // normalize the probabilities...
    #pragma omp parallel for num_threads ( OMP_NUM_THREADS ) reduction ( +:r ) schedule ( auto )
    for ( i = 0; i < n; i++ ) {
        r += p [ i ]; }
    #pragma omp parallel for num_threads ( OMP_NUM_THREADS ) schedule ( auto )
    for ( i = 0; i < n; i++ ) {
        p [ i ] /= r; }

    r = 1.0f / n;

    int lrge [ len ];
    int smll [ len ];

    for ( i = 0; i < n; i++ ) {

        if ( p [ i ] > r ) {
               lrge [ l++ ] = i; }
        else { smll [ s++ ] = i; }
    }

    len += 1;

    vs->prob = ( float * ) malloc ( len );
    vs->alias = ( int * ) malloc ( len );

    while ( s > 0 && l > 0 ) {

        i = smll [ --s ];
        j = lrge [ --l ];

        vs->prob  [ i ] = n * p [ i ];
        vs->alias [ i ] = j;

        p [ j ] += p [ i ] - r;

        if ( p [ j ] > r ) {
               lrge [ l++ ] = j; }
        else { smll [ s++ ] = j; }
    }

    while ( s > 0 ) {
        vs->prob [ smll [ --s ] ] = 1.0f; }
    while ( l > 0 ) {
        vs->prob [ lrge [ --l ] ] = 1.0f; }

    vs->prob [ n ] = vs->prob [ n - 1 ];
    vs->alias [ n ] = vs->alias [ n - 1 ];

    len = strlen ( id );
    vs->id = ( char * ) malloc ( len + 1 );
    strcpy_s ( vs->id, len + 1, id );

    HASH_ADD_KEYPTR ( hh, vose_pool, vs->id, len, vs );

    return vs;
}


int get_vose ( char *id ) {

    float u;
    int i;
    vose_t *vs;

    HASH_FIND_STR ( vose_pool, id, vs );

    vs_rng_uniform ( 1, &u, 0.0f, vs->n );
    i = ( int ) u;

    return ( u - ( float ) i ) <= vs->prob [ i ] ? vs->set [ i ] : vs->set [ vs->alias [ i ] ];
}


int get_vose_state ( const void *state, char *id ) {

    float u;
    int i;
    vose_t *vs;

    HASH_FIND_STR ( vose_pool, id, vs );

    vs_rng_uniform_state ( state, 1, &u, 0.0f, vs->n );
    i = ( int ) u;

    return ( u - ( float ) i ) <= vs->prob [ i ] ? vs->set [ i ] : vs->set [ vs->alias [ i ] ];
}


void delete_vose ( char *id ) {

    vose_t *vs;

    HASH_FIND_STR ( vose_pool, id, vs );

    free ( vs->id );
    free ( vs->set );
    free ( vs->prob );
    free ( vs->alias );

    HASH_DEL ( vose_pool, vs );

    free ( vs );
}


int get_vose_ref ( vose_t *vs ) {

    float u;
    int i;

    vs_rng_uniform ( 1, &u, 0.0f, vs->n );
    i = ( int ) u;

    return ( u - ( float ) i ) <= vs->prob [ i ] ? vs->set [ i ] : vs->set [ vs->alias [ i ] ];
}


int get_vose_state_ref ( const void *state, vose_t *vs ) {

    float u;
    int i;

    vs_rng_uniform_state ( state, 1, &u, 0.0f, vs->n );
    i = ( int ) u;

    return ( u - ( float ) i ) <= vs->prob [ i ] ? vs->set [ i ] : vs->set [ vs->alias [ i ] ];
}


void delete_vose_ref ( vose_t *vs ) {

    free ( vs->id );
    free ( vs->set );
    free ( vs->prob );
    free ( vs->alias );

    HASH_DEL ( vose_pool, vs );

    free ( vs );
}
