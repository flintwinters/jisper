package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
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

func flatMapStr(xs []string, fn func(string) []string) []string {
	out := make([]string, 0)
	for _, x := range xs {
		out = append(out, fn(x)...)
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

func readFileContent(baseDir, filename string) (string, bool) {
	return readTextOrNone(filepath.Join(baseDir, filename))
}

func toRel(base, target string) string {
	rel, err := filepath.Rel(base, target)
	if err != nil {
		return target
	}
	return rel
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

func resolveSystemPrompt(config map[string]any) string {
	sysRaw, _ := asNonEmptyStr(config["system_prompt"])
	projPrompt, _ := asNonEmptyStr(config["project"])
	if projPrompt != "" {
		return sysRaw + "\n\n" + projPrompt
	}
	return sysRaw
}

func resolveUserTask(config map[string]any, routineName string) string {
	if t, ok := resolveRoutineTask(config, routineName); ok {
		return t
	}
	t, _ := asNonEmptyStr(config["task"])
	return t
}

func coerceInt(v any) *int {
	switch x := v.(type) {
	case float64:
		i := int(x)
		if float64(i) == x {
			return &i
		}
		return nil
	case int:
		i := x
		return &i
	case json.Number:
		i64, err := x.Int64()
		if err != nil {
			return nil
		}
		i := int(i64)
		return &i
	case string:
		s := strings.TrimSpace(x)
		if s == "" {
			return nil
		}
		n := json.Number(s)
		i64, err := n.Int64()
		if err != nil {
			return nil
		}
		i := int(i64)
		return &i
	default:
		return nil
	}
}

