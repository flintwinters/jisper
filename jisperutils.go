package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

func asNonEmptyStr(v any) (string, bool) {
	s, ok := v.(string)
	if !ok {
		return "", false
	}
	s = strings.TrimSpace(s)
	if s == "" {
		return "", false
	}
	return s, true
}

func asListOfNonEmptyStr(v any) []string {
	xs, ok := v.([]any)
	if !ok {
		ss, ok := v.([]string)
		if !ok {
			return []string{}
		}
		return filterEmpty(ss)
	}
	out := make([]string, 0, len(xs))
	for _, x := range xs {
		if s, ok := asNonEmptyStr(x); ok {
			out = append(out, s)
		}
	}
	return out
}

func filterEmpty(ss []string) []string {
	out := make([]string, 0, len(ss))
	for _, s := range ss {
		if t := strings.TrimSpace(s); t != "" {
			out = append(out, t)
		}
	}
	return out
}

func dedupeKeepOrder(xs []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(xs))
	for _, x := range xs {
		if x == "" || seen[x] {
			continue
		}
		seen[x] = true
		out = append(out, x)
	}
	return out
}

func readTextOrNone(path string) (string, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	return string(b), true
}

func resolvePathsAndGlobs(values []string, baseDir string) []string {
	one := func(v string) []string {
		s := strings.TrimSpace(v)
		if s == "" {
			return []string{}
		}
		p := filepath.Join(baseDir, s)
		st, err := os.Stat(p)
		if err == nil && st.IsDir() {
			ents, _ := os.ReadDir(p)
			out := make([]string, 0)
			for _, e := range ents {
				if !e.IsDir() {
					out = append(out, filepath.Join(s, e.Name()))
				}
			}
			sort.Strings(out)
			return out
		}
		if err == nil && !st.IsDir() {
			return []string{s}
		}
		matches, _ := filepath.Glob(p)
		out := make([]string, 0)
		for _, m := range matches {
			if mst, _ := os.Stat(m); mst != nil && !mst.IsDir() {
				out = append(out, toRel(baseDir, m))
			}
		}
		sort.Strings(out)
		return out
	}
	out := make([]string, 0)
	for _, v := range values {
		out = append(out, one(v)...)
	}
	return dedupeKeepOrder(out)
}

func buildJinjaContext(cfg map[string]any, source, task, sys string) map[string]any {
	m := map[string]any{"source_text": source, "task": task, "system_prompt": sys}
	for k, v := range cfg {
		m[k] = v
	}
	return m
}

func render(tmpl string, ctx map[string]any) string {
	for k, v := range ctx {
		tmpl = strings.ReplaceAll(tmpl, "{{"+k+"}}", fmt.Sprintf("%v", v))
	}
	return tmpl
}

func toRel(base, target string) string {
	rel, err := filepath.Rel(base, target)
	if err != nil {
		return target
	}
	return rel
}
