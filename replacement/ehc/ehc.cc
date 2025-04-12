#include "ehc.h"

#include <algorithm>
#include <cassert>
#include <random>
#include <utility>

#include "champsim.h"

// Constructor - sets up DRRIP with EHC extension
drrip::drrip(CACHE* cache)
    : replacement(cache), NUM_SET(cache->NUM_SET), NUM_WAY(cache->NUM_WAY), rrpv(static_cast<std::size_t>(NUM_SET * NUM_WAY)),
      access_counters(static_cast<std::size_t>(NUM_SET * NUM_WAY), 0)
{
  // Set up sampler sets for DRRIP
  std::size_t TOTAL_SDM_SETS = NUM_CPUS * NUM_POLICY * SDM_SIZE;
  std::generate_n(std::back_inserter(rand_sets), TOTAL_SDM_SETS, std::knuth_b{1});
  std::sort(std::begin(rand_sets), std::end(rand_sets));
  std::fill_n(std::back_inserter(PSEL), NUM_CPUS, typename decltype(PSEL)::value_type{0});

  // Initialize hit history table
  hit_history_table.resize(NUM_CPUS);
  for (uint32_t cpu = 0; cpu < NUM_CPUS; cpu++) {
    hit_history_table[cpu].resize(HHT_SIZE);
    for (std::size_t i = 0; i < HHT_SIZE; i++) {
      hit_history_table[cpu][i].valid = false;
      hit_history_table[cpu][i].lru_recency = 0;
      hit_history_table[cpu][i].tag = 0;
      hit_history_table[cpu][i].hit_counts.clear();
    }
  }

  bip_counter = 0;
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
 * Helper function to calculate the set index for the hit history table
 * Uses the lower bits of the address after removing the block offset
 */
std::size_t drrip::get_hht_set(champsim::address addr)
{
  // Use lower bits of address for indexing
  return (addr.get_line_addr() & (HHT_SIZE - 1));
}

/*
 * Helper function to determine which way in the set to use for HHT
 * This is a simplified implementation, in a real system you might
 * use more address bits for better distribution
 */
unsigned drrip::get_hht_assoc(champsim::address addr)
{
  // Simple hash function for way selection
  return ((addr.get_line_addr() >> 6) & (HHT_ASSOC - 1));
}

/*
 * Find an entry in the hit history table matching the given address
 * If no match is found, returns an iterator to the entry that should be replaced
 */
std::vector<hit_history>::iterator drrip::find_hit_history(uint32_t cpu, champsim::address addr)
{
  std::size_t set = get_hht_set(addr);
  unsigned tag = (addr.get_line_addr() >> 12); // Upper bits as tag

  // Track LRU entry for possible replacement
  std::vector<hit_history>::iterator lru_entry = hit_history_table[cpu].begin() + set;
  unsigned max_lru = 0;

  // Search for matching entry or find LRU entry
  for (std::size_t i = 0; i < HHT_ASSOC; i++) {
    auto entry = hit_history_table[cpu].begin() + set + i;

    if (entry->valid && entry->tag == tag) {
      return entry; // Found matching entry
    }

    if (!entry->valid) {
      return entry; // Found invalid entry to use
    }

    if (entry->lru_recency > max_lru) {
      max_lru = entry->lru_recency;
      lru_entry = entry;
    }
  }

  return lru_entry; // Return LRU entry if no match found
}

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
 * Predicts the expected number of hits for a cache block based on its history
 * Returns a value between 0 and MAX_HIT_COUNT
 */
unsigned drrip::get_expected_hits(uint32_t cpu, champsim::address addr)
{
  auto entry = find_hit_history(cpu, addr);

  // No history available
  if (!entry->valid || entry->hit_counts.empty()) {
    return 0;
  }

  // Calculate average hit count
  unsigned total_hits = 0;
  unsigned count = 0;

  for (auto hit_count : entry->hit_counts) {
    total_hits += hit_count;
    count++;
  }

  // TODO: Consider weighted average for recency
  unsigned avg_hits = (count > 0) ? (total_hits / count) : 0;
  return std::min(avg_hits, MAX_HIT_COUNT);
}

/*
 * Updates the hit history table with the observed hit count for a cache block
 */
void drrip::update_hit_history(uint32_t cpu, champsim::address addr, unsigned hits)
{
  auto entry = find_hit_history(cpu, addr);

  // Update entry
  entry->valid = true;
  entry->tag = (addr.get_line_addr() >> 12);

  // Update LRU data
  std::size_t set = get_hht_set(addr);
  for (std::size_t i = 0; i < HHT_ASSOC; i++) {
    auto other_entry = hit_history_table[cpu].begin() + set + i;
    if (other_entry->valid) {
      other_entry->lru_recency++;
    }
  }
  entry->lru_recency = 0;

  // Add hit count to history
  entry->hit_counts.push_back(hits);

  // Keep only 4 most recent records
  while (entry->hit_counts.size() > 4) {
    entry->hit_counts.pop_front();
  }
}

/*
 * Main function to update the replacement state on cache accesses
 * Steps:
 * 1. For writebacks, set RRPV to maxRRPV-1 and return
 * 2. For cache hits, increment hit counter and set RRPV to 0
 * 3. For cache misses, initialize hit counter and set RRPV based on expected hits
 *
 * This implements set dueling to adaptively choose between BIP and SRRIP
 * Now also updates hit count tracking for EHC
 */

// Handles replacement state updates when cache is accessed
// Updates RRPV values and tracks hit counters
void drrip::update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip,
                                     champsim::address victim_addr, access_type type, uint8_t hit)
{
  // Handle writebacks
  if (access_type{type} == access_type::WRITE) {
    get_rrpv(set, way) = maxRRPV - 1;
    return;
  }

  // Handle cache hit
  if (hit) {
    increment_access_counter(set, way);
    get_rrpv(set, way) = 0; // Move to MRU position
    return;
  }

  // Handle cache miss
  reset_access_counter(set, way);

  // Find if this is a leader set
  auto begin = std::next(std::begin(rand_sets), triggering_cpu * NUM_POLICY * SDM_SIZE);
  auto end = std::next(begin, NUM_POLICY * SDM_SIZE);
  auto leader = std::find(begin, end, set);

  // Get expected hit count for this address
  unsigned expected_hits = get_expected_hits(triggering_cpu, full_addr);

  // Adjust insertion policy based on expected hits
  if (expected_hits == 0) {
    // Traditional DRRIP - no expected hits
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
  } else {
    // EHC enhancement - set RRPV based on expected hits
    // More expected hits = lower RRPV (less likely to be evicted)
    get_rrpv(set, way) = std::max(0u, maxRRPV - expected_hits);

    // Still update policy selection for learning
    if (leader == begin) {
      PSEL[triggering_cpu]--;
    } else if (leader == std::next(begin)) {
      PSEL[triggering_cpu]++;
    }
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
 * Now enhanced with EHC to consider expected hit counts
 */
long drrip::find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                        champsim::address full_addr, access_type type)
{
  auto begin = std::next(std::begin(rrpv), set * NUM_WAY);
  auto end = std::next(begin, NUM_WAY);

  // Find blocks with highest RRPV
  std::vector<size_t> max_rrpv_indices;
  unsigned max_value = 0;

  // First find the maximum RRPV value
  for (auto it = begin; it != end; ++it) {
    if (*it > max_value) {
      max_value = *it;
    }
  }

  // Then collect all blocks with that RRPV
  for (auto it = begin; it != end; ++it) {
    if (*it == max_value) {
      max_rrpv_indices.push_back(std::distance(begin, it));
    }
  }

  // Use expected hit count as tiebreaker
  long victim_way = 0;
  if (max_rrpv_indices.size() > 1) {
    // Multiple candidates - use hit count prediction to choose
    unsigned min_expected_hits = MAX_HIT_COUNT + 1;

    for (auto idx : max_rrpv_indices) {
      champsim::address block_addr = champsim::address(current_set[idx].address);
      unsigned expected_hits = get_expected_hits(triggering_cpu, block_addr);

      if (expected_hits < min_expected_hits) {
        min_expected_hits = expected_hits;
        victim_way = idx;
      }
    }
  } else if (!max_rrpv_indices.empty()) {
    // Only one candidate - use it
    victim_way = max_rrpv_indices[0];
  } else {
    // Shouldn't happen - fallback to traditional approach
    auto victim = std::max_element(begin, end);
    if (auto rrpv_update = maxRRPV - *victim; rrpv_update != 0)
      for (auto it = begin; it != end; ++it)
        *it += rrpv_update;

    victim_way = std::distance(begin, victim);
  }

  // Record hit count of victim for future prediction
  if (current_set[victim_way].valid) {
    champsim::address victim_addr = champsim::address(current_set[victim_way].address);
    unsigned hits = get_access_counter(set, victim_way);
    update_hit_history(triggering_cpu, victim_addr, hits);
  }

  return victim_way;
}

/*
 * Increment the access counter for a specific cache block
 */
void drrip::increment_access_counter(long set, long way)
{
  std::size_t index = static_cast<std::size_t>(set * NUM_WAY + way);

  if (access_counters[index] < MAX_HIT_COUNT) {
    access_counters[index]++;
  }
}

/*
 * Reset the access counter for a specific cache block to zero
 */
void drrip::reset_access_counter(long set, long way)
{
  std::size_t index = static_cast<std::size_t>(set * NUM_WAY + way);
  access_counters[index] = 0;
}

/*
 * Get the current access count for a specific cache block
 */
unsigned drrip::get_access_counter(long set, long way)
{
  std::size_t index = static_cast<std::size_t>(set * NUM_WAY + way);
  return access_counters[index];
}
