# SpiceLab OpenSpec Changes

Este diretório contém todas as propostas de mudanças (change proposals) para o desenvolvimento do SpiceLab, organizadas seguindo a metodologia OpenSpec.

## 📋 Visão Geral

As mudanças foram organizadas de acordo com os marcos (milestones) M7-M20 do roadmap do projeto, além de melhorias de usabilidade cross-cutting.

## 🎯 Change Proposals Criadas

### M7 - Measurement Library (Biblioteca de Medidas)
**Diretório:** `m7-measurement-library/`

Implementa biblioteca completa de medidas para análise de resultados de simulação:
- Medidas AC: magnitude, fase, GBW, PM, GM, fT
- Medidas TRAN: overshoot, settling time, rise/fall time, SNR, THD, ENOB
- Medidas NOISE: input-referred noise, PSD, NF
- Utilitários DSP: windowing, FFT, filtros

**Specs:** `measurement/`

---

### M8 - Model Management (Gerenciamento de Modelos)
**Diretório:** `m8-model-management/`

Sistema de gerenciamento de modelos SPICE, subcircuitos e Verilog-A:
- Registry de modelos com versionamento
- Resolução de paths e caching
- Suporte a Verilog-A via ADMS
- Integração com writers de netlist

**Specs:** `models/`, `verilog-a/`

---

### M9 - Optimization & DOE (Otimização e Design de Experimentos)
**Diretório:** `m9-optimization-doe/`

Ferramentas de otimização e exploração de espaço de design:
- Otimizadores SciPy + plugins (Nevergrad, Optuna)
- Análise de sensitividade (finite differences + Xyce .sens)
- DOE: Full Factorial, Latin Hypercube, Sobol
- Multi-objetivo com frentes de Pareto

**Specs:** `optimization/`, `sensitivity/`, `doe/`

---

### M10 - Performance & I/O (Performance e Zero-Copy)
**Diretório:** `m10-performance-io/`

Otimizações de performance e I/O para datasets grandes:
- Leitura mmap de arquivos binários .raw
- Lazy loading e acesso por chunks
- Export Parquet com compressão colunar
- Cache content-addressable otimizado

**Specs:** `io-performance/`

---

### M11 - UX Enhancements (Melhorias de Experiência do Usuário)
**Diretório:** `m11-ux-enhancements/`

Ferramentas para melhorar a experiência do usuário:
- Widgets interativos para Jupyter
- CLI `spicelab doctor` para diagnóstico
- Helpers de visualização (Bode, Nyquist, step)
- Documentação MkDocs completa
- Mensagens de erro melhoradas

**Specs:** `widgets/`, `cli/`, `docs/`, `viz/`

---

### M12 - Plugin System (Sistema de Plugins)
**Diretório:** `m12-plugin-system/`

Sistema de plugins e estabilidade de API:
- Entry points para measurements, readers, engines
- Plugin registry com validação
- Política SemVer e deprecação
- Release automation (tag → PyPI)
- Telemetria opcional opt-in

**Specs:** `plugins/`

---

### M13-M20 - Advanced Features (Features Avançadas)
**Diretório:** `m13-m20-advanced-features/`

Roadmap de longo prazo com features enterprise:

- **M13:** IDE extensions (VS Code, JupyterLab), schematic I/O (LTspice, KiCad)
- **M14:** Distributed execution (Ray, Dask, SLURM, K8s), distributed cache
- **M15:** Mixed-signal co-sim (XSPICE, HDL, FMI/FMU), firmware-in-loop
- **M16:** PDK management, corner analysis, parameter extraction
- **M17:** Yield analysis, robust design, multi-objective optimization
- **M18:** Performance hardcore (indexed RAW, Arrow/Parquet pipeline)
- **M19:** Quality & security (regression tests, SBOM, i18n)
- **M20:** Ecosystem (plugin marketplace, templates, commercial licensing)

**Specs:** `ide-integration/`, `distributed/`, `mixed-signal/`, `pdk-management/`, `yield-analysis/`

---

### Usability Improvements (Melhorias de Usabilidade)
**Diretório:** `usability-improvements/`

Melhorias cross-cutting de usabilidade:
- Templates de circuitos comuns
- Smart unit parsing (1k, 1kohm, 1e3 → todos válidos)
- Validação de circuito (floating nodes, shorts)
- Workflow shortcuts (quick_ac, method chaining)
- Mensagens de erro aprimoradas
- Debugging tools (dry-run, verbose mode)
- Type hints completos
- Biblioteca de patterns reutilizáveis

**Specs:** `usability/`

---

## 🗂️ Estrutura de Cada Change

Cada change proposal segue o formato OpenSpec:

```
m[N]-feature-name/
├── proposal.md          # Why, What Changes, Impact
├── tasks.md             # Implementation checklist
├── design.md            # Technical decisions (optional)
└── specs/               # Delta specifications
    └── [capability]/
        └── spec.md      # ADDED/MODIFIED/REMOVED requirements
```

## 🚀 Próximos Passos

### Para Implementação

1. **Review & Approval:** Revisar proposals e aprovar prioridades
2. **Sequência Sugerida:**
   - M7 (measurements) - alto valor, baixa complexidade
   - M8 (models) - necessário para designs reais
   - M9 (optimization) - fecha ciclo de design
   - M10 (performance) - needed para escala
   - M11 (UX) - melhora adoção
   - M12 (plugins) - prepara ecossistema
   - Usability improvements - incremental em todos os marcos
   - M13-M20 - roadmap de longo prazo (2-3 anos)

3. **Validação:** Para cada change aprovado:
   ```bash
   openspec validate m7-measurement-library --strict
   openspec validate m8-model-management --strict
   # ... etc
   ```

4. **Implementação:** Seguir tasks.md de cada change sequencialmente

5. **Archive:** Após deployment:
   ```bash
   openspec archive m7-measurement-library --yes
   ```

### Para Validação Agora

Execute validação de todas as specs:

```bash
openspec list                           # Lista todas as changes
openspec validate --strict              # Valida tudo em modo strict
```

## 📊 Estatísticas

- **Total de Changes:** 8 (M7-M12 individuais + M13-M20 consolidado + Usability)
- **Total de Specs:** ~20 capability specs
- **Total de Requirements:** ~150+ requirements detalhados
- **Total de Scenarios:** ~300+ scenarios de teste
- **Estimated Timeline:**
  - M7-M12 + Usability: 12-18 meses
  - M13-M20: 24-36 meses adicionais

## 🎯 Prioridades Recomendadas

### 🔥 Alta Prioridade (Próximos 6 meses)
1. **M7** - Measurements (completa análise pós-simulação)
2. **Usability** - Improvements (reduz fricção imediata)
3. **M11** - UX/Doctor (facilita adoção)

### ⚡ Média Prioridade (6-12 meses)
4. **M8** - Models (suporte a PDKs reais)
5. **M9** - Optimization (ciclo de design completo)
6. **M10** - Performance (necessário para escala)

### 🔮 Longo Prazo (12+ meses)
7. **M12** - Plugins (ecossistema)
8. **M13-M20** - Advanced features (enterprise)

## 💡 Melhorias Adicionadas

Além dos marcos do update.md original, foram adicionadas melhorias de usabilidade que facilitam significativamente o uso:

- **Smart defaults:** Templates e presets para workflows comuns
- **Better errors:** Mensagens contextuais com sugestões de ação
- **Type safety:** Type hints completos para melhor IDE support
- **Shortcuts:** APIs encurtadas para tarefas comuns
- **Validation:** Checks preventivos antes de simular
- **Debugging:** Ferramentas para troubleshooting rápido

## 📚 Referências

- [OpenSpec Methodology](./AGENTS.md) - Guia completo do processo
- [Project Context](./project.md) - Convenções do projeto
- [update.md](../update.md) - Roadmap original em português

## 🤝 Contribuindo

Para adicionar novos changes:

1. Escolha um `change-id` único (kebab-case, verb-led)
2. Crie estrutura: `mkdir -p openspec/changes/[change-id]/{specs/[capability]}`
3. Escreva proposal.md, tasks.md, specs/
4. Valide: `openspec validate [change-id] --strict`
5. Solicite review antes de implementar

---

**Status:** ✅ Todas as specs criadas e prontas para validação
**Última Atualização:** 2025-10-27
**Próxima Ação:** Executar `openspec validate --strict` e revisar proposals
