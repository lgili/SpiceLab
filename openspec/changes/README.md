# SpiceLab OpenSpec Changes

This directory contains detailed specifications for all planned improvements to SpiceLab, organized into 3 phases over 24 months.

## 📁 Structure

```
changes/
├── phase1-foundations/     # Months 1-6: Stability, Performance, DX
│   ├── m1-stability/       ✅ Complete specification
│   ├── m2-performance/     ✅ Complete specification
│   ├── m3-components/      ✅ Complete specification
│   ├── m4-dx/              ✅ Complete specification
│   ├── m5-docs/            ✅ Complete specification
│   └── m6-community/       ✅ Complete specification
├── phase2-advanced/        # Months 7-12: Advanced Analytics, Extensibility
│   ├── m7-measurements/    ✅ Complete specification
│   ├── m8-models/          ✅ Complete specification
│   ├── m9-optimization/    ✅ Complete specification
│   ├── m10-io/             ✅ Complete specification
│   ├── m11-ux/             ✅ Complete specification
│   └── m12-plugins/        ✅ Complete specification
├── phase3-enterprise/      # Months 13-18: Enterprise Features, Industry Adoption
│   ├── m13-mixed-signal/   ✅ Complete specification
│   ├── m14-pdk/            ✅ Complete specification
│   ├── m15-distributed/    ✅ Complete specification
│   ├── m16-yield/          ✅ Complete specification
│   ├── m17-schematic/      ✅ Complete specification
│   └── m18-enterprise/     ✅ Complete specification
├── usability-improvements/ 🔄 In Progress (partially implemented)
└── archive/                📦 Archived/completed changes
```

**Total:** 18 milestones, 36 files (proposal + tasks each), ~250KB of specifications

---

## 🚀 Quick Start

### 1. Understand the Vision
Read these in order:
1. [`../../TRANSFORMATION_SUMMARY.md`](../../TRANSFORMATION_SUMMARY.md) - Executive summary (5 min read)
2. [`../ROADMAP.md`](../ROADMAP.md) - Complete 24-month plan (15 min read)
3. [`../../CRITICAL_ANALYSIS.md`](../../CRITICAL_ANALYSIS.md) - Deep competitive analysis (30 min read)

### 2. Pick a Milestone
Choose based on priority and your interests:
- **Want to start now?** → M1 (Stability) or M2 (Performance)
- **Like working with data?** → M7 (Measurements) or M8 (Models)
- **UX enthusiast?** → M4 (DX) or M11 (UX Enhancements)
- **Performance nerd?** → M2 (Performance) or M10 (I/O Overhaul)

### 3. Read the Proposal
```bash
# Example: M1 Stability
cat phase1-foundations/m1-stability/proposal.md
```

### 4. Check the Tasks
```bash
# See weekly breakdown
cat phase1-foundations/m1-stability/tasks.md
```

### 5. Start Implementing
Follow the implementation plan, check off tasks as you go!

---

## 📊 Priority Guide

### 🔴 Start Immediately (Critical Path)
These form the foundation - nothing else can proceed without them:

1. **[M1: Stability & Testing](phase1-foundations/m1-stability/proposal.md)**
   - 95% coverage, property tests, fuzzing
   - **Why critical:** Prevents regressions, enables confident refactoring
   - **Duration:** 4-6 weeks
   - **Dependencies:** None

2. **[M2: Performance Baseline](phase1-foundations/m2-performance/proposal.md)**
   - 10x faster, benchmarks, profiling
   - **Why critical:** Performance is a feature, sets baseline
   - **Duration:** 4-6 weeks
   - **Dependencies:** M1 (for stress tests)

3. **[M3: Component Library](phase1-foundations/m3-components/proposal.md)**
   - 100+ components, 50+ vendor models
   - **Why critical:** Can't use library without components
   - **Duration:** 4-6 weeks
   - **Dependencies:** None

### 🟠 High Priority (Months 4-8)
Significantly improve usability and capabilities:

4. **[M4: Developer Experience](phase1-foundations/m4-dx/proposal.md)**
   - Better errors, validation, templates
   - **Duration:** 3-4 weeks
   - **Dependencies:** M1

5. **[M7: Measurements](phase2-advanced/m7-measurements/proposal.md)**
   - 30+ specs (gain, bandwidth, THD, etc.)
   - **Duration:** 8-10 weeks
   - **Dependencies:** M1, M3

6. **[M8: Model Management](phase2-advanced/m8-models/proposal.md)**
   - 500+ vendor models, Verilog-A, IBIS
   - **Duration:** 10-12 weeks
   - **Dependencies:** M3, M7

### 🟡 Medium Priority (Months 6-12)
Enhance ecosystem and adoption:

7. **[M5: Documentation](phase1-foundations/m5-docs/proposal.md)**
   - Tutorial, examples, API reference
   - **Duration:** 4-5 weeks
   - **Dependencies:** M1-M4 (need stable APIs)

8. **[M6: Community](phase1-foundations/m6-community/proposal.md)**
   - Discord, guidelines, automation
   - **Duration:** 3-4 weeks
   - **Dependencies:** M5 (docs needed for contributors)

9. **[M9: Optimization](phase2-advanced/m9-optimization/proposal.md)**
   - scipy, DEAP, DoE, sensitivity
   - **Duration:** 8-10 weeks
   - **Dependencies:** M1, M2

10. **[M10: I/O Performance](phase2-advanced/m10-io/proposal.md)**
    - 100x faster, streaming, GPU
    - **Duration:** 10-12 weeks
    - **Dependencies:** M2 (baseline)

11. **[M11: UX Enhancements](phase2-advanced/m11-ux/proposal.md)**
    - VSCode, Web UI, Jupyter magics
    - **Duration:** 10-12 weeks
    - **Dependencies:** M1, M4

12. **[M12: Plugin System](phase2-advanced/m12-plugins/proposal.md)**
    - Extensibility, marketplace
    - **Duration:** 8-10 weeks
    - **Dependencies:** M1, M3

### ⚪ Lower Priority (Months 13-18+)
Enterprise and advanced features:

13. **[M13: Mixed-Signal](phase3-enterprise/m13-mixed-signal/proposal.md)**
14. **[M14: PDK Integration](phase3-enterprise/m14-pdk/proposal.md)**
15. **[M15: Distributed](phase3-enterprise/m15-distributed/proposal.md)**
16. **[M16: Yield Analysis](phase3-enterprise/m16-yield/proposal.md)**
17. **[M17: Schematic](phase3-enterprise/m17-schematic/proposal.md)**
18. **[M18: Enterprise](phase3-enterprise/m18-enterprise/proposal.md)**

---

## 📋 What's In Each Milestone?

Every milestone folder contains:

### `proposal.md` (20-40KB)
Comprehensive technical specification:
- **Problem Statement** - What's broken/missing and why it matters
- **Objectives** - Specific, measurable goals (SMART criteria)
- **Technical Design** - Architecture, code examples, algorithms
- **Implementation Plan** - Weekly breakdown
- **Success Metrics** - Must have, should have, nice to have
- **Risks & Mitigations** - What could go wrong and how to prevent it
- **Dependencies** - Prerequisites from other milestones
- **References** - Standards, papers, documentation

### `tasks.md` (4-12KB)
Actionable implementation guide:
- **Phased Breakdown** - Tasks organized by week
- **Checkboxes** - Track progress as you work
- **Estimated Time** - Duration per phase
- **Acceptance Criteria** - Definition of done
- **Testing Checklist** - Quality gates before marking complete
- **Dependencies** - Blocking relationships with other milestones

---

## 🎯 Success Metrics

### Overall (24 Months)
- [ ] 99% test coverage (from ~60%)
- [ ] 1,000+ components (from ~30)
- [ ] 500+ vendor models (from 0)
- [ ] 100x performance improvement
- [ ] 5,000 GitHub stars (from ~10)
- [ ] 200k PyPI downloads/month (from <100)
- [ ] 100+ contributors (from 1)
- [ ] 100+ companies using (from 0)

### Phase 1 Completion (Month 6)
- [ ] 95% test coverage
- [ ] 100+ components
- [ ] 50+ vendor models
- [ ] 10x performance improvement
- [ ] 500 GitHub stars
- [ ] 5k PyPI downloads/month
- [ ] 10+ contributors

### Phase 2 Completion (Month 12)
- [ ] 98% test coverage
- [ ] 300+ components
- [ ] 200+ vendor models
- [ ] 50x performance improvement
- [ ] 2,000 GitHub stars
- [ ] 50k PyPI downloads/month
- [ ] 30+ contributors

---

## 🗺️ Dependency Graph

```
Phase 1 (Foundations):
  M1 ─┬─→ M2 ──→ M3 ──┬─→ M4
      │              └─→ M5
      └─────────────────→ M6

Phase 2 (Advanced):
  M1 + M3 ──→ M7 ────┬─→ M8
  M1 + M2 ──→ M9     │
  M2 ────────→ M10   │
  M1 + M4 ──→ M11    │
  M1 + M3 ──→ M12    │
                     └──→ (Phase 3)

Phase 3 (Enterprise):
  M8 + M12 ──→ M13
  M8 + M12 ──→ M14
  M10 ─────→ M15
  M9 + M15 ──→ M16
  M3 + M4 ──→ M17
  M6 + M12 ──→ M18
```

**Key insight:** M1 (Stability) is foundational for almost everything. Start there!

---

## 🛠️ Implementation Workflow

### Step 1: Prepare
```bash
# Clone repo
git clone https://github.com/lgili/circuit_toolkit.git
cd circuit_toolkit

# Create feature branch
git checkout -b feature/m1-stability

# Install dev dependencies
uv sync --all-extras --dev
```

### Step 2: Understand
```bash
# Read proposal (understand the "why")
cat openspec/changes/phase1-foundations/m1-stability/proposal.md

# Read tasks (understand the "how")
cat openspec/changes/phase1-foundations/m1-stability/tasks.md

# Check dependencies
grep -r "Dependencies" openspec/changes/phase1-foundations/m1-stability/
```

### Step 3: Implement
```bash
# Work through tasks week by week
# Check off items in tasks.md as you complete them
vim openspec/changes/phase1-foundations/m1-stability/tasks.md

# Run tests frequently
pytest tests/ -v

# Check coverage
pytest --cov=spicelab --cov-report=html
```

### Step 4: Validate
Before marking milestone complete:
- [ ] All tasks checked off
- [ ] All acceptance criteria met
- [ ] Tests pass (including new tests)
- [ ] Coverage gates met (90%+ for module)
- [ ] Docs updated (if public API changed)
- [ ] CHANGELOG.md updated
- [ ] MyPy strict passes
- [ ] Ruff linting passes

### Step 5: Submit
```bash
# Commit changes
git add .
git commit -m "feat(core): implement M1 stability milestone

- Add 50+ property tests with Hypothesis
- Implement fuzzing for parser (20 targets)
- Stress tests up to 10k components
- Replace assertions with exceptions
- Achieve 95% coverage

Closes #123"

# Push and create PR
git push origin feature/m1-stability
gh pr create --fill
```

### Step 6: Complete
After PR is merged:
```bash
# Update proposal status
echo -e "\n---\n**Status:** Completed ($(date +%Y-%m-%d))" >> \
  openspec/changes/phase1-foundations/m1-stability/proposal.md

# Optionally archive
mkdir -p openspec/changes/archive/
mv openspec/changes/phase1-foundations/m1-stability \
   openspec/changes/archive/
```

---

## 📚 Learning Resources

### For Contributors
- **New to Testing?** Read M1 proposal for property-based testing intro
- **New to Performance?** Read M2 proposal for profiling techniques
- **New to SPICE?** Read `openspec/project.md` for domain context
- **New to OpenSpec?** Read `openspec/AGENTS.md` for methodology

### External Resources
- [Hypothesis Docs](https://hypothesis.readthedocs.io/) - Property-based testing
- [py-spy](https://github.com/benfred/py-spy) - Sampling profiler
- [PyO3 Guide](https://pyo3.rs/) - Python-Rust bindings
- [xarray Tutorial](https://tutorial.xarray.dev/) - N-D labeled arrays

---

## 🤝 How to Contribute

### Claiming a Milestone
1. Check [GitHub Issues](https://github.com/lgili/circuit_toolkit/issues) for milestone
2. Comment "I'd like to work on this"
3. Wait for assignment (or self-assign if you're a contributor)
4. Fork repo and follow implementation workflow above

### Reporting Issues with Specs
Found a problem with a proposal?
1. Open issue with label `openspec-revision`
2. Link to the proposal file
3. Explain the issue (technical error, missing info, etc.)
4. Suggest a fix if possible

### Adding New Milestones
Have an idea for M19 or M20?
1. Create proposal following template
2. Add to appropriate phase directory
3. Link dependencies
4. Submit PR for review

---

## ❓ FAQ

**Q: Can I work on milestones out of order?**
A: Check dependencies first! Some milestones require others (e.g., M7 needs M1 and M3).

**Q: How long does each milestone take?**
A: Phase 1: 3-6 weeks. Phase 2: 8-12 weeks. Phase 3: 10-12 weeks. Varies by team size and experience.

**Q: Can I work on multiple milestones simultaneously?**
A: Yes, if they're independent! E.g., M1 and M3 can be done in parallel. See dependency graph.

**Q: What if I can only contribute part-time?**
A: That's fine! Break milestones into smaller chunks. Even 1-2 tasks per week adds up.

**Q: Do I need to complete ALL tasks in tasks.md?**
A: "Must have" criteria are required. "Should have" are strongly recommended. "Nice to have" are optional.

**Q: How do I know when I'm done?**
A: All acceptance criteria in `tasks.md` are met, tests pass, docs updated, PR approved and merged.

---

## 📊 Current Status

### Completed
- ✅ All 18 milestone specifications written
- ✅ All proposals and tasks finalized
- ✅ Dependency graph established
- ✅ Roadmap documentation complete

### In Progress
- 🔄 [Usability Improvements](usability-improvements/) (partially implemented)

### Next Actions
1. Start M1 (Stability & Testing)
2. Recruit contributors for M2 and M3 (parallel work)
3. Setup CI infrastructure (needed for M1)
4. Launch Discord community (M6 prep)

---

## 📞 Contact

**Maintainer:** Luiz Carlos Gili (@lgili)
**Repository:** https://github.com/lgili/circuit_toolkit
**Documentation:** https://lgili.github.io/CircuitToolkit/
**Discord:** TBD (launching in M6)

---

**Last Updated:** 2025-01-19
**Document Version:** 2.0 (completely reorganized)
**Status:** Ready for execution
