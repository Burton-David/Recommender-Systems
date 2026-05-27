// Go recommendation service over the same model.json the Python service reads.
//
// Loads the model at startup, holds it as plain float32 slices, serves
// GET /recommend?user_id=<id>&n=<count> with one matrix-vector multiply per
// request. Deliberately uses stdlib only (encoding/json, net/http, math) —
// pulling in gonum would essentially reproduce Python's BLAS path and
// muddy the comparison.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
)

type model struct {
	userIDs       []int64
	itemIDs       []int64
	userIndex     map[int64]int
	userFactors   [][]float32 // (n_users, n_factors)
	itemFactorsT  [][]float32 // (n_items, n_factors)
	nFactors      int
}

type modelJSON struct {
	UserIDs       []int64     `json:"user_ids"`
	ItemIDs       []int64     `json:"item_ids"`
	UserFactors   [][]float32 `json:"user_factors"`
	ItemFactorsT  [][]float32 `json:"item_factors_T"`
}

func loadModel(path string) (*model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var raw modelJSON
	if err := json.NewDecoder(f).Decode(&raw); err != nil {
		return nil, fmt.Errorf("decode %s: %w", path, err)
	}
	if len(raw.UserFactors) == 0 || len(raw.ItemFactorsT) == 0 {
		return nil, errors.New("model has empty factor matrices")
	}
	idx := make(map[int64]int, len(raw.UserIDs))
	for i, uid := range raw.UserIDs {
		idx[uid] = i
	}
	return &model{
		userIDs:      raw.UserIDs,
		itemIDs:      raw.ItemIDs,
		userIndex:    idx,
		userFactors:  raw.UserFactors,
		itemFactorsT: raw.ItemFactorsT,
		nFactors:     len(raw.UserFactors[0]),
	}, nil
}

// scoreUser computes (n_items,) scores for the given user index. The
// inner loop is the cost center; everything else is bookkeeping.
func (m *model) scoreUser(uIdx int) []float32 {
	u := m.userFactors[uIdx]
	scores := make([]float32, len(m.itemFactorsT))
	for i, row := range m.itemFactorsT {
		var s float32
		for k := 0; k < m.nFactors; k++ {
			s += row[k] * u[k]
		}
		scores[i] = s
	}
	return scores
}

func topN(scores []float32, n int) []int {
	order := make([]int, len(scores))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool { return scores[order[i]] > scores[order[j]] })
	if n > len(order) {
		n = len(order)
	}
	return order[:n]
}

func recommendHandler(m *model) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		userID, err := strconv.ParseInt(r.URL.Query().Get("user_id"), 10, 64)
		if err != nil {
			http.Error(w, "bad user_id", http.StatusBadRequest)
			return
		}
		n, _ := strconv.Atoi(r.URL.Query().Get("n"))
		if n <= 0 {
			n = 10
		}
		uIdx, ok := m.userIndex[userID]
		if !ok {
			http.Error(w, "unknown user_id", http.StatusNotFound)
			return
		}
		scores := m.scoreUser(uIdx)
		order := topN(scores, n)
		items := make([]int64, len(order))
		for i, idx := range order {
			items[i] = m.itemIDs[idx]
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"items": items})
	}
}

func main() {
	modelPath := flag.String("model", "../model.json", "path to model.json")
	port := flag.Int("port", 8001, "HTTP port")
	flag.Parse()

	m, err := loadModel(*modelPath)
	if err != nil {
		log.Fatalf("load model: %v", err)
	}
	log.Printf("loaded model: %d users, %d items, k=%d", len(m.userIDs), len(m.itemIDs), m.nFactors)

	http.HandleFunc("/recommend", recommendHandler(m))
	http.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"status":"ok"}`))
	})

	addr := fmt.Sprintf(":%d", *port)
	log.Printf("listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}
