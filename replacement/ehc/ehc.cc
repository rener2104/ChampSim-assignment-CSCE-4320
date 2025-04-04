#include <algorithm>
#include <cassert>
#include <random>
#include <utility>

#include "champsim.h"
#include "drrip.h"

/*
 * Constructor for DRRIP (Dynamic Re-Reference Interval Prediction) cache replacement policy
 * Steps:
 * 1. Initialize base class and member variables from the cache parameters
 * 2. Create a vector to store RRPV values for each cache block (NUM_SET * NUM_WAY total blocks)
 * 3. Calculate the total number of set dueling monitor (SDM) sets needed
 * 4. Randomly select cache sets to be used for policy evaluation (leader sets)
 * 5. Sort the randomly selected sets for easier lookup
 * 6. Initialize the policy selector (PSEL) counters to zero for each CPU
 */
drrip::drrip(CACHE* cache) : replacement(cache), NUM_SET(cache->NUM_SET), NUM_WAY(cache->NUM_WAY), rrpv(static_cast<std::size_t>(NUM_SET * NUM_WAY))
{
  // randomly selected sampler sets
  std::size_t TOTAL_SDM_SETS = NUM_CPUS * NUM_POLICY * SDM_SIZE;
  std::generate_n(std::back_inserter(rand_sets), TOTAL_SDM_SETS, std::knuth_b{1});
  std::sort(std::begin(rand_sets), std::end(rand_sets));
  std::fill_n(std::back_inserter(PSEL), NUM_CPUS, typename decltype(PSEL)::value_type{0});
}

/*
 * Helper function to access the RRPV of a specific cache block
 * Steps:
 * 1. Calculate the linear index in the rrpv vector from the set and way
 * 2. Return a reference to the RRPV value at that index
 *
 * Note: RRPV (Re-Reference Prediction Value) indicates how soon a block is
 * likely to be referenced again
 */
unsigned& drrip::get_rrpv(long set, long way) { return rrpv.at(static_cast<std::size_t>(set * NUM_WAY + way)); }

/*
 * Implements the Bimodal Insertion Policy (BIP)
 * Steps:
 * 1. Set the RRPV to maxRRPV (distant re-reference)
 * 2. Increment the BIP counter
 * 3. If the counter reaches BIP_MAX, reset it and set RRPV to maxRRPV-1 (this happens infrequently)
 *
 * BIP helps with scan-resistant behavior by occasionally setting lower RRPV values
 */
void drrip::update_bip(long set, long way)
{
  get_rrpv(set, way) = maxRRPV;

  bip_counter++;
  if (bip_counter == BIP_MAX) {
    bip_counter = 0;
    get_rrpv(set, way) = maxRRPV - 1;
  }
}

/*
 * Implements the Static Re-Reference Interval Prediction (SRRIP) policy
 * Steps:
 * 1. Set RRPV to maxRRPV-1 (intermediate re-reference prediction)
 *
 * SRRIP is effective for mixed access patterns and provides protection
 * against thrashing
 */
void drrip::update_srrip(long set, long way) { get_rrpv(set, way) = maxRRPV - 1; }

/*
 * Main function to update the replacement state on cache accesses
 * Steps:
 * 1. For writebacks, set RRPV to maxRRPV-1 and return
 * 2. For cache hits, set RRPV to 0 (recently used) and return
 * 3. For cache misses:
 *    a. Find if the current set is a leader set by checking rand_sets
 *    b. If it's a follower set, use either BIP or SRRIP based on PSEL counter
 *    c. If it's a BIP leader set, decrement PSEL and apply BIP
 *    d. If it's an SRRIP leader set, increment PSEL and apply SRRIP
 *
 * This implements set dueling to adaptively choose between BIP and SRRIP
 */
void drrip::update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                     champsim::address victim_addr, access_type type, uint8_t hit)
{
  // do not update replacement state for writebacks
  if (access_type{type} == access_type::WRITE) {
    get_rrpv(set, way) = maxRRPV - 1;
    return;
  }

  // cache hit
  if (hit) {
    get_rrpv(set, way) = 0; // for cache hit, DRRIP always promotes a cache line to the MRU position
    return;
  }

  // cache miss
  auto begin = std::next(std::begin(rand_sets), triggering_cpu * NUM_POLICY * SDM_SIZE);
  auto end = std::next(begin, NUM_POLICY * SDM_SIZE);
  auto leader = std::find(begin, end, set);

  if (leader == end) { // follower sets
    auto selector = PSEL[triggering_cpu];
    if (selector.value() > (selector.maximum / 2)) { // follow BIP
      update_bip(set, way);
    } else { // follow SRRIP
      update_srrip(set, way);
    }
  } else if (leader == begin) { // leader 0: BIP
    PSEL[triggering_cpu]--;
    update_bip(set, way);
  } else if (leader == std::next(begin)) { // leader 1: SRRIP
    PSEL[triggering_cpu]++;
    update_srrip(set, way);
  }
}

/*
 * Function to find a victim block for replacement
 * Steps:
 * 1. Find the range of RRPV values for the given set
 * 2. Find the block with the maximum RRPV (least likely to be referenced soon)
 * 3. If the max RRPV is less than maxRRPV, "age" all blocks by incrementing their RRPVs
 *    (this ensures eventual replacement even if no block has maxRRPV)
 * 4. Verify the victim is within valid range and return its way index
 *
 * This implements the age-based victim selection mechanism
 */
long drrip::find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                        champsim::address full_addr, access_type type)
{
  // look for the maxRRPV line
  auto begin = std::next(std::begin(rrpv), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);

  auto victim = std::max_element(begin, end);
  if (auto rrpv_update = maxRRPV - *victim; rrpv_update != 0)
    for (auto it = begin; it != end; ++it)
      *it += rrpv_update;

  assert(begin <= victim);
  assert(victim < end);
  return std::distance(begin, victim); // cast protected by assertions
}
