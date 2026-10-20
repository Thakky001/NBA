---
name: omniroute
description: "AI gateway and router across 300+ AI providers and models with auto-fallback, rate-limit bypass, and token compression (RTK/Caveman). Use when routing LLM requests, configuring OpenAI-compatible endpoints at http://localhost:20128/v1, managing providers, or reducing token costs."
license: MIT
---

# OmniRoute: Multi-Provider AI Gateway & Router

OmniRoute is an open-source, local AI gateway and proxy that unifies access to 300+ AI models and providers through a single OpenAI-compatible endpoint (`http://localhost:20128/v1`). It features quota-aware auto-fallback, latency optimization, and native prompt compression (RTK + Caveman).

---

## 1. Quick Start & Server Management

### Start OmniRoute Server
```bash
# Using npx or global CLI
npx omniroute serve

# Background daemon mode
npx omniroute serve -d

# Check server health and status
curl -s http://localhost:20128/health
```

Default Gateway URL:
- **Base URL:** `http://localhost:20128/v1`
- **Dashboard:** `http://localhost:20128/dashboard`

---

## 2. Core Capabilities

### A. Dynamic Model Routing & Fallback Combos
When a primary model hits a rate limit (HTTP 429), budget cap, or connection failure, OmniRoute automatically cascades down a configured fallback chain without failing the client request.

Supported Combo Strategies:
- `priority`: Try provider A, fallback to provider B, then provider C.
- `round-robin`: Balance load evenly across equivalent models.
- `lowest-latency`: Direct prompts to the provider with the lowest p50 response time.
- `lowest-cost`: Prioritize free-tier models and lowest price-per-token providers.
- `auto-combo`: Intelligent dynamic routing based on prompt complexity and provider quotas.

### B. Prompt & Token Compression
OmniRoute incorporates RTK and Caveman compression filters:
- **RTK (Command Output Compression):** Strips redundant terminal formatting, build noise, and repeated stack traces (saving 50–90% input tokens).
- **Caveman (Prose Compression):** Compresses system instructions and conversation history while preserving semantic intent.

### C. Unified API Endpoints
All client calls target standard OpenAI endpoints:
```bash
curl http://localhost:20128/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer omniroute-local" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## 3. CLI Subcommands

| Command | Description |
|---|---|
| `omniroute serve` | Start/stop/restart the local proxy gateway server |
| `omniroute providers` | Add, list, test, and manage API keys for upstream providers |
| `omniroute models` | Query available models and provider catalog |
| `omniroute routing` | Configure fallback chains, routing strategies, and combos |
| `omniroute keys` | Generate and manage client API keys and rate limits |
| `omniroute resilience` | View circuit-breaker states, cooldowns, and error rates |

---

## 4. Agent Guidelines

When operating within an OmniRoute-enabled environment:
1. **Prefer `auto` or configured combo aliases** over hardcoding a single fragile provider.
2. **Handle 429 Gracefully:** Rely on OmniRoute's auto-fallback rather than adding custom retry loops.
3. **Use Local Compression:** Enable RTK on large command outputs before piping to LLM context.
