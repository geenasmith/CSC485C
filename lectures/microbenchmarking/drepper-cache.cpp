/**
 */

#include <iostream>  // std::cout
#include <cassert>	 // assert()

#include "timing.hpp"
#include "data-generation1.hpp"


struct calc_sum
{
	template < typename T >
		auto operator () ( std::vector< T > data ) const
            -> typename std::remove_reference< decltype( data.front().pad[ 0 ] ) >::type
		{
			assert( "Not empty" && data.size() > 0 );
			auto sum = 0u;

			std::for_each(std::cbegin(data), std::cend(data), [&sum](auto const& x){ sum = sum + x.next;});

            return sum;
		}
};

// Observe that we are now taking command line arguments so that we can run this with different options
// rather than recompiling for every test.
int main( int argc, char **argv )
{
	auto const num_tests  = 1000u; // Number of random trials to test out
    auto const padding = 64u; // effectively defining the size of a node
			// each node is "padding" * size_t -> size_t is 64 bits. 
			// 32*64 bits in each node.
    auto const list_length = 1u << 12; // # of nodes. 1u -> 00000000 00000001 << 10 = 00000010 00000000 
	// total size of the "working data set" is 32*64* 2^10

    // Note that I have "linearised" the 2d array: i.e., instead of creating a T[][] c-style 2d array
    // like in the Doumler slides, I continue to use a 1d std::vector() and I will just use offsets
    // (e.g., array[ i * width + j]) to index into cell (i,j).
	auto const test_cases = csc586::benchmark::rand_vec_of_drepper_lists< uint32_t, padding, list_length >( num_tests );
	
	// not elegant, but this switches which implementation to run depending on whether a third argument was given
	auto const run_time   = csc586::benchmark::benchmark( calc_sum{}
														, test_cases );

    std::cout << "Average time (us): " << run_time << std::endl;

	return 0;
}
