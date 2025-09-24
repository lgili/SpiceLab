# Core Contracts (M1)

> Estado: Concluído (baseline estável após migração para Pydantic v2)

Este documento descreve as interfaces **estáveis** estabelecidas no Marco M1 do projeto:
- Especificação de análises (`AnalysisSpec`)
- Especificação de varreduras (`SweepSpec`)
- Probes / pontos de medição (`Probe`)
- Metadados e hashing determinístico (`ResultMeta`, `stable_hash`, `circuit_hash`)
- Normalizadores de entrada (`ensure_analysis_spec`, `ensure_sweep_spec`)

## Objetivos de M1
1. Modelo de dados validado e serializável (Pydantic) para análises e sweeps.
2. Hash determinístico para cache e reprodutibilidade.
3. Infraestrutura de resultados com metadados coerentes entre engines.
4. Extensão clara (pronta para M2/M3: múltiplos engines e orquestração de sweeps/MC).

---
## AnalysisSpec
```python
AnalysisSpec(mode: Literal["op","dc","ac","tran","noise"], args: dict[str, Any])
```
Regras:
- `args` aceita apenas primitivos (`str|int|float|bool|None`) ou listas/tuplas desses.
- Exemplos:
  ```python
  AnalysisSpec("op", {})
  AnalysisSpec("tran", {"tstep": "1u", "tstop": "5m"})
  AnalysisSpec("ac", {"n": 100, "fstart": 10, "fstop": 1e6})
  ```
- Mapeamento para directives é feito em cada engine (`_spec_to_directives`).

## SweepSpec
```python
SweepSpec(variables: dict[str, list[float|str]])
```
Normalização:
- Cada valor pode ser número ou string com sufixo de engenharia (`1k`, `2.2u`, `47n`).
- Lista não pode ser vazia.
- Após validação, `variables[name]` torna-se `list[float]` (SI puro).

Exemplo:
```python
sweep = SweepSpec({"R": ["1k", "2k", 3000]})
assert sweep.variables["R"] == [1000.0, 2000.0, 3000.0]
```

## Probe
```python
Probe(kind: Literal["voltage","current"], target: str)
```
- `voltage`: `target` é nome de nó (`"out"`)
- `current`: `target` é referência de componente (`"R1"`, `"VDD"` ...)
- Validação garante alvo não vazio.

Integração atual:
- `run_simulation(..., probes=[Probe(kind="voltage", target="out")])`
- Armazenado em `ResultMeta.probes` e replicado em `DatasetResultHandle.attrs()['probes']`.
- Engines ainda não filtram sinais no netlist (fase futura: seleção de colunas / restrição de export).

## Hashing determinístico
Funções:
- `stable_hash(obj)` → sha1 curto (12 hex) de uma representação canônica JSON
- `circuit_hash(circuit, extra={...})` → inclui `build_netlist()` + metadados opcionais

Objetivo: chave de cache, deduplicação e rastreabilidade de resultados.

Garantias:
- Ordenação de dicionários padronizada.
- Pydantic models → `model_dump()` ordenado.
- Sets → listas ordenadas.
- Objetos com `build_netlist()` → string do netlist.

## ResultMeta & DatasetResultHandle
`ResultMeta` campos:
- `engine`, `engine_version`, `netlist_hash`
- `analyses: list[AnalysisSpec]`
- `probes: list[Probe]`
- `attrs: dict[str, Any]` (informações livres adicionais, p.ex. caminhos de log)

`DatasetResultHandle`:
- `dataset()` → `xarray.Dataset`
- `to_pandas()` / `to_polars()` (conversão lazy)
- `attrs()` → visão serializável dos metadados (analyses/probes serializados)

## Normalizadores
Para flexibilidade de input externo (ex: JSON/YAML):
```python
ensure_analysis_spec({"mode": "op", "args": {}})  # -> AnalysisSpec
ensure_sweep_spec({"variables": {"R": [1,2,3]}})    # -> SweepSpec
```

## Quebra de compatibilidade (Migração dataclass → Pydantic)
- Antigo dataclass `AnalysisSpec` substituído por Pydantic.
- Construtor posicional preservado: `AnalysisSpec("ac", {...})` continua válido.
- Código que dependia de `__dict__` deve usar `model_dump()` ou atributos diretos.

## Extensões planejadas (M2/M3+)
- Filtragem/extração orientada a `Probe` (reduzir colunas carregadas).
- Suporte a `noise` (sintaxe engine específica).
- Orquestração de varreduras multi-dimensionais (`SweepSpec`) multiplicando análises.
- Cache inteligente chaveado por `circuit_hash + analysis hash + sweep point`.

## Exemplos rápidos
```python
from spicelab.engines.orchestrator import run_simulation
from spicelab.core.types import AnalysisSpec, Probe

analyses = [AnalysisSpec("op"), AnalysisSpec("ac", {"n": 100, "fstart": 10, "fstop": 1e6})]
probes = [Probe(kind="voltage", target="out")]
res = run_simulation(circuit, analyses, probes=probes, engine="ngspice")

print(res.attrs()["analyses"])  # lista de dicts
print(res.attrs()["probes"])    # lista de dicts
```

## Test Coverage
Cenários cobertos:
- Normalização de `SweepSpec`
- Rejeição de valores inválidos
- Hash determinístico independente de ordenação
- Probes: validação de alvo vazio

Pendentes (recomendado adicionar):
- Serialização de `probes` confirmada em `DatasetResultHandle.attrs()`
- Erros de sintaxe faltante em cada modo (mensagens dedicadas)

---
Última atualização: 2025-09-24
