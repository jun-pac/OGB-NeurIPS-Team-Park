#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#include "sample_cpu.h"
#include "utils.h"

#ifdef _WIN32
#include <process.h>
#endif

// Returns `rowptr`, `col`, `n_id`, `e_id`
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> 
sample_adj_cpu(torch::Tensor rowptr, torch::Tensor col, torch::Tensor idx,
               int64_t num_ppr_neighbors, int64_t num_atr_neighbors, int64_t num_ins_neighbors, 
               bool replace, torch::Tensor relation_ptr){
// Add "relation bound" argument
// relation ptr is (N*3+1,) **rowptr-like** tensor 
// i.e. 3*i+1 is rowptr of paper, 3*i+2 is rowptr of author, 3*i+3 is rowptr of institution.
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  CHECK_CPU(idx);
  CHECK_CPU(relation_ptr); // OIOCHA
  CHECK_INPUT(idx.dim() == 1);

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto idx_data = idx.data_ptr<int64_t>();
  auto relation_ptr_data=relation_ptr.data_ptr<int64_t>();

  auto out_rowptr = torch::empty({idx.numel() + 1}, rowptr.options());
  auto out_rowptr_data = out_rowptr.data_ptr<int64_t>();
  out_rowptr_data[0] = 0;

  std::vector<std::vector<std::tuple<int64_t, int64_t>>> cols; // col, e_id
  std::vector<int64_t> n_ids;
  std::unordered_map<int64_t, int64_t> n_id_map;

  int64_t i;
  for (int64_t n = 0; n < idx.numel(); n++) {
    i = idx_data[n];
    cols.push_back(std::vector<std::tuple<int64_t, int64_t>>());
    n_id_map[i] = n;
    n_ids.push_back(i);
  }

  int64_t n, c, e, row_start, row_end, row_count, ppr_count, atr_count, ins_count;

  if (replace) { // Sample with replacement ===============================

    for (int64_t i = 0; i < idx.numel(); i++) {
      n = idx_data[i];
      row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
      row_count = row_end - row_start;

      if (row_count > 0) {
        for (int64_t j = 0; j < num_neighbors; j++) {
          e = row_start + uniform_randint(row_count);
          c = col_data[e];

          if (n_id_map.count(c) == 0) {
            n_id_map[c] = n_ids.size();
            n_ids.push_back(c);
          }
          cols[i].push_back(std::make_tuple(n_id_map[c], e));
        }
      }
      out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
    }

  } else { // Sample without replacement via Robert Floyd algorithm ============

    for (int64_t i = 0; i < idx.numel(); i++) {
        n = idx_data[i];
        row_start = rowptr_data[n], row_end = rowptr_data[n + 1];
        row_count = row_end - row_start;
        ppr_count=relation_ptr_data[3*n+1]-relation_ptr_data[3*n];
        atr_count=relation_ptr_data[3*n+2]-relation_ptr_data[3*n+1];
        ins_count=relation_ptr_data[3*n+3]-relation_ptr_data[3*n+2];
        std::unordered_set<int64_t> perm;

        // Sample ppr
        if(ppr_count <= num_ppr_neighbors){
            for(int64_t j = 0; j < ppr_count; j++)
                perm.insert(j);
            } 
        }
        else {
            for (int64_t j = ppr_count - num_ppr_neighbors; j < ppr_count; j++) {
                if (!perm.insert(uniform_randint(j)).second)
                perm.insert(j);
            }
        }

        // Sample atr
        if(atr_count <= num_atr_neighbors){
            for(int64_t j = 0; j < atr_count; j++)
                perm.insert(ppr_count+j);
            } 
        }
        else {
            for (int64_t j = atr_count - num_atr_neighbors; j < atr_count; j++) {
                if (!perm.insert(ppr_count+uniform_randint(j)).second)
                perm.insert(ppr_count+j);
            }
        }

        // Sample ins
        if(ins_count <= num_ins_neighbors){
            for(int64_t j = 0; j < ins_count; j++)
                perm.insert(ppr_count+atr_count+j);
            } 
        }
        else {
            for (int64_t j = ins_count - num_ins_neighbors; j < ins_count; j++) {
                if (!perm.insert(ppr_count+atr_count+uniform_randint(j)).second)
                perm.insert(ppr_count+atr_count+j);
            }
        }

        for (const int64_t &p : perm) {
        e = row_start + p;
        c = col_data[e]; // As this is csr format, c is real idx of node.

        if (n_id_map.count(c) == 0) {
            // count is same as find here.
            n_id_map[c] = n_ids.size();
            // n_id_map is unordered_map : n_idx's value -> n_idx's internal idx
            // don't increase its size 
            n_ids.push_back(c);
            // vector initially same as n_idx
        }
        cols[i].push_back(std::make_tuple(n_id_map[c], e));
        // cols : vector<vector<tuple<int,int>>>, first dimension has len(n_idx) size
        // Store col, e_id information
        // n_id_map[c] is sampled node's pseudo idx, e is sampled node's real idx (order of selected edge in TOTAL edges.) 
        // I just have understood that e_id is real EDGE index, NOT real NODE index.
        }
        out_rowptr_data[i + 1] = out_rowptr_data[i] + cols[i].size();
        // Generating new rowptr.
        // Of course it interact with pseudo idx.
        // I think out_rowptr_data is ptr of out_rowptr
    }
  }

  int64_t N = n_ids.size();
  auto out_n_id = torch::from_blob(n_ids.data(), {N}, col.options()).clone();

  int64_t E = out_rowptr_data[idx.numel()]; // Total size of sampled edges. 
  auto out_col = torch::empty({E}, col.options());
  auto out_col_data = out_col.data_ptr<int64_t>();
  auto out_e_id = torch::empty({E}, col.options());
  auto out_e_id_data = out_e_id.data_ptr<int64_t>();

  i = 0;
  for (std::vector<std::tuple<int64_t, int64_t>> &col_vec : cols) {
    std::sort(col_vec.begin(), col_vec.end(),
              [](const std::tuple<int64_t, int64_t> &a,
                 const std::tuple<int64_t, int64_t> &b) -> bool {
                return std::get<0>(a) < std::get<0>(b);
              });
    for (const std::tuple<int64_t, int64_t> &value : col_vec) {
      out_col_data[i] = std::get<0>(value);
      out_e_id_data[i] = std::get<1>(value);
      i += 1;
    }
  }

  return std::make_tuple(out_rowptr, out_col, out_n_id, out_e_id);
}