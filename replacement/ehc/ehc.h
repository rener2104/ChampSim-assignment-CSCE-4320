#ifndef REPLACEMENT_EHC_H
#define REPLACEMENT_EHC_H

#include <array>
#include <deque>
#include <vector>

#include "cache.h"
#include "modules.h"
#include "msl/fwcounter.h"

/*
 * Structure to track cache line access history
 * Used by Expected Hit Count (EHC) extension to DRRIP
 * Components:
 * 1. valid - Indicates if this entry contains valid data
 * 2. lru_recency - Used for replacement policy within the hit history table
 * 3. tag - Address tag to identify the cache line this entry tracks
 * 4. hit_counts - Queue to store sequence of hit counts for prediction
 */
struct hit_history {
  bool valid;
  unsigned lru_recency;
  unsigned tag;
  std::deque<unsigned> hit_counts;

  // Needed to create a default constructor for the hit_history table
  hit_history() : valid(false), lru_recency(0), tag(0), hit_counts() {}
};

/*
 * EHC (Expected Hit Count) implementation
 * Built upon DRRIP (Dynamic Re-Reference Interval Prediction)
 * Inherits from the base replacement policy module
 */
struct drrip : public champsim::modules::replacement {
private:
  /*
   * Helper method to access the RRPV of a specific cache block
   * Parameters:
   * - set: The cache set index
   * - way: The way index within the set
   * Returns: Reference to the RRPV value for the specified block
   */
  unsigned& get_rrpv(long set, long way);

  // helper function to calculate the set index for the hit history table
  // parameters
  //-addr: memory address
  // returns set index in the hit history table
  std::size_t get_hht_set(champsim::address addr);

  unsigned get_hht_assoc(champsim::address addr);

  std::vector<hit_history>::iterator find_hit_history(uint32_t cpu, champsim::address addr);

  // Access counter for each cache block to track hits
  std::vector<unsigned> access_counters;

public:
  /*
   * Constants for DRRIP algorithm configuration:
   * 1. maxRRPV: Maximum Re-Reference Prediction Value (distant future reference)
   * 2. NUM_POLICY: Number of policies used in set dueling (BIP and SRRIP)
   * 3. SDM_SIZE: Set Dueling Monitor size (number of sets per policy per CPU)
   * 4. BIP_MAX: Counter value for Bimodal Insertion Policy frequency
   * 5. PSEL_WIDTH: Bit width of the Policy Selector counter
   */
  static constexpr unsigned maxRRPV = 3;
  static constexpr std::size_t NUM_POLICY = 2;
  static constexpr std::size_t SDM_SIZE = 32;
  static constexpr unsigned BIP_MAX = 32;
  static constexpr unsigned PSEL_WIDTH = 10;

  /*
   * Constants for EHC (Expected Hit Count) extension:
   * 1. HHT_SIZE: Size of the Hit History Table per CPU core
   * 2. HHT_ASSOC: Associativity of the Hit History Table
   * 3. MAX_HIT_COUNT: Maximum value for hit counters
   */
  static constexpr std::size_t HHT_SIZE = 2048; // 2K entries per core
  static constexpr std::size_t HHT_ASSOC = 16;  // 16-way associativ
  static constexpr unsigned MAX_HIT_COUNT = 7;  // 3-bit counter max value

  /*
   * Cache geometry parameters:
   * - NUM_SET: Number of sets in the cache
   * - NUM_WAY: Associativity of the cache (ways per set)
   */
  long NUM_SET, NUM_WAY;

  /*
   * State variables for DRRIP algorithm:
   * 1. bip_counter: Counter for BIP frequency control
   * 2. rand_sets: Vector of randomly selected sets for policy evaluation
   * 3. PSEL: Policy Selector counters (one per CPU)
   * 4. rrpv: Vector storing RRPV values for each cache block
   */
  unsigned bip_counter;
  std::vector<std::size_t> rand_sets;
  std::vector<champsim::msl::fwcounter<PSEL_WIDTH>> PSEL;
  std::vector<unsigned> rrpv;

  /*
   * EHC specific data structures:
   * - hit_history_table: Table to track access patterns for each address
   *   Organized as one table per CPU core
   */
  std::vector<std::vector<hit_history>> hit_history_table; //[cpu][set][way]

  /*
   * Constructor for initializing the DRRIP replacement policy
   * Parameters:
   * - cache: Pointer to the cache object this policy manages
   */
  drrip(CACHE* cache);

  /*
   * Core replacement policy interface functions:
   *
   * 1. find_victim: Selects a cache block for replacement
   *    - Identifies the block with the highest RRPV value
   *    - Parameters include CPU ID, instruction ID, set index, and more
   *    - Returns the way index of the selected victim
   *
   * 2. update_replacement_state: Updates RRPV values on cache access
   *    - Called on every cache hit and miss
   *    - Implements the set dueling mechanism to choose between BIP and SRRIP
   *    - Updates RRPV based on access type and policy decisions
   */
  long find_victim(uint32_t triggering_cpu, uint64_t instr_id, long set, const champsim::cache_block* current_set, champsim::address ip,
                   champsim::address full_addr, access_type type);
  void update_replacement_state(uint32_t triggering_cpu, long set, long way, champsim::address full_addr, champsim::address ip, champsim::address victim_addr,
                                access_type type, uint8_t hit);

  /*
   * Policy-specific helper methods:
   *
   * 1. update_bip: Implements Bimodal Insertion Policy
   *    - Usually sets RRPV to maxRRPV with occasional exceptions
   *
   * 2. update_srrip: Implements Static Re-Reference Interval Prediction
   *    - Sets RRPV to maxRRPV-1 for new entries
   */
  void update_bip(long set, long way);
  void update_srrip(long set, long way);

  /*
   * EHC specific methods:
   *
   * 1. get_expected_hits: Predicts how many hits a cache line will receive
   *    - Uses hit history to make predictions
   *    - Parameters: CPU ID and memory address
   *    - Returns: Predicted number of hits
   *
   * 2. update_hit_history: Records actual hit counts for future prediction
   *    - Updates the hit history table with new observations
   *    - Parameters: CPU ID, address, and number of hits observed
   *
   * 3. increment_access_counter: Increments the hit counter for a specific cache block
   *    - Parameters: set and way indices of the block
   *
   * 4. reset_access_counter: Resets the hit counter for a specific cache block
   *    - Parameters: set and way indices of the block
   *
   * 5. get_access_counter: Returns the current hit count for a specific cache block
   *    - Parameters: set and way indices of the block
   *    - Returns: Number of hits for the block
   */
  unsigned get_expected_hits(uint32_t cpu, champsim::address addr);
  void update_hit_history(uint32_t cpu, champsim::address addr, unsigned hits);
  void increment_access_counter(long set, long way);
  void reset_access_counter(long set, long way);
  unsigned get_access_counter(long set, long way);
};

#endif
