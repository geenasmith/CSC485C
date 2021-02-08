/**
 * A small application to generate random input vectors for benchmarking tests.
 */

#include <random> 	// std::rand, std::srand, std::default_random_engine

namespace csc586 {
namespace benchmark {

// Node in a linked list (using name `l` to match Drepper)
template < size_t NPAD >
	struct l
	{
		uint32_t next;		// pointer to next node
		size_t pad[ NPAD ]; // don't care what this data is
	};

// Wrapper for a contiguous set of nodes
template < size_t NPAD >
	using node_set = std::vector< l< NPAD > >;


// Initialises a contiguously allocated linked list
template < size_t NPAD, size_t LENGTH >
	node_set< NPAD > make_list()
	{
        node_set< NPAD > l( LENGTH );

		for( auto i = 0u; i < LENGTH - 1; ++i )
		{
			l[ i ].next = i + 1;
		}
		l[ LENGTH - 1 ].next = 0;

		return l;
	}

/**
 * Creates and returns a vector pre-populated by repeatedly calling gen size times.
 *
 * see timing.hpp for a description of functors (e.g., gen). size_t is the built-in
 * type that refers to the size of things. On most architectures, it is a 64-bit
 * unsigned integer.
 *
 * Note the use of auto with a trailing return type. This can be a useful pattern
 * when the return type depends on a generic input parameter.
 */
template < typename RandomGenerator >
	auto build_rand_vec( RandomGenerator gen, size_t const size )
        -> std::vector< decltype((gen)()) >
	{
		// see timing.hpp for a description of decltype().
		// Initialises a vector with the appropriate size, in which we will create the random data.
		std::vector< decltype((gen)()) > random_data( size );

		// std::generate() = STL. This could be done with a loop from 0,...,size, calling gen() each
		// iteration. See https://www.fluentcpp.com/2017/01/05/the-importance-of-knowing-stl-algorithms/
		// for a compelling discussion of why using STL algorithms like this is better than explicit loops.
		// Spoiler: it better self-documents (in code) intent and is less prone to off-by-one-errors.
		// For an explanation of syntax, see the use of std::for_each() in timing.hpp
		std::generate( random_data.begin(), random_data.end(), gen );

		return random_data;
	}


/**
 * Builds a vector of vectors of uniformly generated random primitive values of type T
 */
template < typename T >
    auto uniform_rand_vec_of_vec( size_t const n, size_t const m )
        -> std::vector< std::vector< T > >
    {
        return build_rand_vec( [ m ](){
            return build_rand_vec( 
	                  [](){ return static_cast< T >( std::rand() ); }
	                , m ); }
            , n );
    }


/**
 * Builds a vector of Drepper-style linked lists of uniformly generated random primitive values of type T
 */
template < typename T, size_t NPAD, size_t LIST_LENGTH >
    auto rand_vec_of_drepper_lists( size_t const n )
        -> std::vector< node_set< NPAD > >
    {
        return build_rand_vec( [](){ return make_list< NPAD, LIST_LENGTH >(); }
	                         , n );
    }

} // namespace benchmark
} // namespace csc586
