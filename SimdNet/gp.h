#ifndef __GP_H__
#define __GP_H__

#include <windows.h>
#include <stdio.h>
#include <stddef.h>
#include <iso646.h>
#define _USE_MATH_DEFINES
#include <imath.h>
#include <float.h>
#include <omp.h>
#include <malloc.h>
#include <mkl_vsl.h>
#include <uthash.h>
#include <stdbool.h>
#include <xmmintrin.h>
#include <time.h>
#include <urlmon.h>
#include <malloc_nD.h>
#include <librandom.h>
#include <gpsort.hpp>
#include <gp_types.h>
#include <run.h>
#include "stack.h"
#include "timsort.h"


#define GP_VERSION 1


#ifndef OMP_NUM_THREADS
#define OMP_NUM_THREADS atoi ( getenv ( "OMP_NUM_THREADS" ) )
#endif
#define OMP_LOAD 6

#define NRLE_TABLE_GRID_SIZE 10UL
#define MAX_PERCENTAGE_SELECTIVE_PRESSURE 200
// below periods are [10, 100)
#define MIN_PERIOD 10
#define MAX_PERIOD 100
#define MIN_DAYS_EVALUATED 1200
#define MAX_INSTRUCTIONS 16384

#define SALES_MARGIN 5.0
#define SALES_MARKUP ((float)(SALES_MARGIN/100.0+1.0))
#define SALES_MARKDOWN ((float)(100.0/(SALES_MARGIN+100.0)))

#define EMPTY -1            // Indicate stack empty

#define SELL_COST (0.99f) // The net return after cost
#define INV_SELL_COST ((float)(1.0/(double)SELL_COST))
#define BUY_COST (1.01f)
#define INV_BUY_COST ((float)(1.0/(double)BUY_COST))

#define MINIMUM_ORDER_SIZE 0.025f

#define ROUND_ABS(a) (llroundf(fabsf(a)))
#define S_SIGN(x) ((__int32)(((x)>(-0.5f))-((x)<0.5f))) // Stepped sign. -1, 0, 1
#define A_SIGN(x) (abs((__int32)(((x)>(-0.5f))-((x)<0.5f)))) // Stepped sign. 0, 1
#define SIGN(x) (-((signbitf((x))<<1)-1)) // -1 or 1

#define MIN(a,b) ((a)>(b)?(b):(a))
#define MAX(a,b) ((a)<(b)?(b):(a))
#define SWAP(a,b) {float t=(a); (a)=(b); (b)=t;}
#define ABS(a) ((a)<0?-(a):(a))
#define COMP(i1,i2) (((i1)>(i2))-((i1)<(i2)))
#define SWAP_UINT(a,b) {const unsigned __int32 t=(a); (a)=(b); (b)=t;}
#define GETBIT(x,in) (((in)[((x)>>3)])&(1<<((x)&7)))
#define TOGGLE_BITS(x,y,z) ((x)^=((1ULL<<(y))|(1ULL<<(z)))) // toggle bits number y and z

#define SIZE(object) ((__int32)object[0])
#define INPUT_LENGTH(program) (program[SIZE(program)+1])

#define NEW_LINE printf("\n")

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

extern portfolio_t *portfolio_pool;
extern population_t pop;
extern const ocp_t ocp [ NUMBER_FUNCTIONS ];

extern float *op_price;
extern float *hi_price;
extern float *lo_price;
extern float *cl_price;

extern float *input_data;
extern date_t *date;
extern const char *dow [ 7 ];

// Evolution
vose_t *initiate_comu ( void );
void evolution ( const __int32 number_of_generations );

// Crossover
void crossover_state ( const void *state, const program_t *op1, const program_t *op2, program_t *np1, program_t *np2 );
void crossover ( const program_t *op1, const program_t *op2, program_t *np1, program_t *np2 );
void crossover_eii ( const __int32 *ancestor_id, const __int32 *deceased_id );
void parallel_crossover_state ( const void *state, const program_t *op1, const program_t *op2, program_t *np1, program_t *np2 );
void parallel_crossover ( const program_t *op1, const program_t *op2, program_t *np1, program_t *np2 );
void parallel_crossover_evaluate_insert ( const __int32 *ancestor_id, const __int32 *deceased_id );
void brood_recombination ( const __int32 brood_size, const program_t *op1, const program_t *op2, individual_t *ni1, individual_t *ni2 );
void brood_recombination_eii ( const __int32 *ancestor_id, const __int32 *deceased_id );
void headless_chicken_crossover_state ( const void *state, program_t *op, program_t *np1, program_t *np2 );
void headless_chicken_crossover ( program_t *op, program_t *np1, program_t *np2 );
void headless_chicken_crossover_eii ( const __int32 *ancestor_id, const __int32 *deceased_id );
void headless_chicken_crossover_with_brood_recombination ( __int32 flock_size, __int32 brood_size, program_t *op, individual_t *ni );
void headless_chicken_crossover_with_brood_recombination_eii ( const __int32 *ancestor_id, const __int32 *deceased_id );

// Evaluation
void select_sample_interval ( const void *state, const unsigned __int16 input_length, const __int32 data_per_day, __int32 *day1, __int32 *day2 );
bool fitness_state ( const void *state, program_t *program, float *fitness, __int32 *evaluations );
bool fitness ( __int32 id );
void evaluate_individual ( __int32 id );
void evaluate_fitness_population ( void );
void evaluate_fitness_population_timed ( void );
void evaluate_fitness_population_with_stats ( __int32 *best, float *best_fitness, float *best_length, float *average_fitness, float *average_length );

// Rank
long double solve_non_linear_rank_equation ( unsigned __int32 pop_size, const unsigned __int32 max_percentage_selective_pressure );
unsigned __int32 get_nlre_table_size ( unsigned __int32 size );
float *generate_nlre_table ( const unsigned __int32 pop_size, const unsigned __int32 nlre_table_size );
float non_linear_rank_equation ( const unsigned __int32 max_percentage_selective_pressure );
void parsimonic_rank_fitness_timed ( const unsigned __int32 selective_pressure, __int32 *sus_sample, __int32 *inverse_rws_sample );

// Population
bool set_program_parameters ( unsigned __int16 max_length, unsigned __int16 max_input_length );
bool set_population_parameters ( __int32 pop_size, __int32 sam_size );
bool initiate_population ( bool use_nlr, __int32 pop_size, __int32 sam_size, unsigned __int16 max_length, unsigned __int16 max_input_length );
void set_all_brng_seeds ( void );
bool initiate_from_disk ( const char *fn );
void save_program ( const __int32 id, const char *fn );
void save_n_best_programs ( const __int32 n, const char *fn );
void save_date_stamp_n_best_programs ( const __int32 n );
void save_population ( const char *fn );
void save_date_stamp_population ( void );
void free_population ( void  );
void generate_population ( void );
void generate_population_timed ( void );
void generate_evaluate_population ( void );
void generate_individual_state ( const void *state, const __int32 id );
void evaluate_and_insert_individual ( const __int32 id, program_t program );
void evaluate_pair_and_insert_individual_state ( const void *state, const __int32 id, program_t *program_1, program_t *program_2 );
void insert_individual ( const __int32 id, const individual_t individual );
__int32 population_average_length ( void );
void population_stats ( __int32 *best, float *best_fitness, float *best_length, float *average_fitness, float *average_length );

// Program
vose_t *initiate_opcodes ( void );
void generate_program_state ( const void *state, program_t *program );
void generate_program ( program_t *program );
void parallel_generate_2_programs ( program_t *p1, program_t *p2 );
void free_program ( program_t *program );
void free_individual ( individual_t *individual );
void print_program ( const program_t *program );
void print_program_oc ( const program_t *program );
void print_program_adresses ( program_t *program );

// Support
bool machine_id ( unsigned char *id );
bool almost_equal_2s_complement_float ( float a, float b, unsigned __int32 max_ulps );
bool almost_equal_2s_complement_double ( double a, double b, unsigned __int64 max_ulps );
void msec_to_time ( __int32 msec, __int32 *h, __int32 *m, __int32 *s, __int32 *ms );
unsigned __int64 __int64_to_gray ( unsigned __int64 uinteger );
unsigned __int64 gray_to___int64 ( unsigned __int64 gray );
__int32 omp_num_threads ( void );
date_t today ( void );
void print_date ( date_t date );
__int32 file_exist ( char *fn );
__int32 datestamp ( void );
__int32 datestamp_n_days_ago ( const __int32 n );
void datestamp_s ( char *date_s );
void pause ( float time );
char *fpeeks ( char *line, __int32 n, FILE *file );
bool is_power_of_2 ( unsigned __int32 x );
unsigned __int32 upper_power_of_two ( unsigned __int32 v );
unsigned __int32 upper_multiple_of_np2 ( unsigned __int32 v, unsigned __int32 n );
unsigned __int32 upper_multiple_of_n ( unsigned __int32 v, unsigned __int32 n );
unsigned __int32 upper_divides_n ( unsigned __int32 v, unsigned __int32 n );
unsigned __int32 lower_divides_n ( unsigned __int32 v, unsigned __int32 n );
__int32 rand64_s ( unsigned __int64 *random );
long double bico ( __int32 n, __int32 k );
void surf ( unsigned __int32 out [ 8 ], unsigned __int32 in [ 12 ], unsigned __int32 dex [ 32 ] );
float kahan_sum ( float *input, __int32 n );
unsigned __int32 rand_r ( unsigned __int32 *seed );
void qsort_float ( float *data, __int32 size );

// Data
__int32 init_data_estoxx ( __int32 date_stamp );
__int32 init_data_yahoo ( __int32 date_stamp );
void free_data ( void );
void download_data_yahoo ( char *fn );
void download_data_estoxx ( char *fn );
__int32 days_diff ( __int32 y, __int32 m, __int32 d );
__int32 days_diff1_s ( char *date_s );
__int32 days_diff2_s ( char *date_s1, char *date_s2 );
__int32 is_after_cot ( void );
__int32 day_of_the_week ( __int32 y, __int32 m, __int32 d );
__int32 day_of_the_week_s ( char *date_s );
__int32 today_day_of_the_week ( void );
FILE *fopen_estoxx_file ( char *fn );
FILE *fopen_yahoo_file ( char *fn );

// Portfolio
bool malloc_portfolio_pool ( void );
void free_portfolio_pool ( void );
void portfolio_pool_portfolio_clean ( void );

void portfolio_clean ( portfolio_t *portfolio );
portfolio_t *assign_clean_portfolio ( void );

bool portfolio_add_on_stop_buy_order_qty ( portfolio_t *portfolio, const float qty, const float price );
bool portfolio_add_on_stop_buy_order_ratio ( portfolio_t *portfolio, const float ratio, const float price );
bool portfolio_fill_orders ( portfolio_t *portfolio, const float high_price, const float low_price );

void portfolio_print ( const portfolio_t *portfolio );

// Cuda_sample
__int32 cuda_rws ( const float *raw_fitness, const __int32 size );
void cuda_inverse_rws_sample ( __int32 *inverse_rws_sample );
void cuda_inverse_rws_sample_test ( float *raw_fitness, unsigned __int32 size, __int32 *inverse_rws_sample );
void rank_fitness ( const unsigned __int32 selective_pressure, __int32 *sus_sample, __int32 *inverse_rws_sample );
void parsimonic_rank_fitness ( const unsigned __int32 selective_pressure, __int32 *sus_sample, __int32 *inverse_rws_sample );
void nl_rank_fitness ( const unsigned __int32 selective_pressure, __int32 *sus_sample, __int32 *inverse_rws_sample );
void parsimonic_nl_rank_fitness ( const unsigned __int32 selective_pressure, __int32 *sus_sample, __int32 *inverse_rws_sample );
void cuda_n_fitest ( unsigned __int32 n, __int32 *n_fitest );

// Random
void malloc_stream_pool_ ( void );
void free_stream_pool_ ( void );
VSLStreamStatePtr *stream_state_ ( void );

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // end __GP_H__
