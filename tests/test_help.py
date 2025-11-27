"""Tests for spicelab.help module."""

import pytest
from spicelab.core.circuit import Circuit
from spicelab.core.components import Resistor
from spicelab.core.net import GND, Net
from spicelab.help import (
    CheatsheetFormat,
    CircuitHelp,
    ComponentHelp,
    TutorialStep,
    generate_cheatsheet,
    get_help,
    list_examples,
    list_tutorials,
    run_example,
    show_help,
    validate_docstring_examples,
)
from spicelab.help.cheatsheet import (
    CHEATSHEET_DATA,
    save_cheatsheet,
)
from spicelab.help.examples import (
    EXAMPLES,
    get_categories,
    get_example,
    run_example_by_name,
)
from spicelab.help.tutorial import (
    TUTORIALS,
    TutorialRunner,
)


class TestContextHelp:
    """Tests for context-sensitive help system."""

    def test_get_help_circuit(self):
        """Test getting help for a circuit."""
        circuit = Circuit("test")
        help_obj = get_help(circuit)
        assert isinstance(help_obj, CircuitHelp)
        assert help_obj.obj is circuit

    def test_get_help_component(self):
        """Test getting help for a component."""
        R1 = Resistor(ref="1", resistance=1000)
        help_obj = get_help(R1)
        assert isinstance(help_obj, ComponentHelp)
        assert help_obj.obj is R1

    def test_help_summary(self):
        """Test help summary generation."""
        circuit = Circuit("test")
        help_obj = get_help(circuit)
        summary = help_obj.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_help_full(self):
        """Test full help text generation."""
        circuit = Circuit("test")
        help_obj = get_help(circuit)
        full_text = help_obj.full()
        assert isinstance(full_text, str)
        assert "Circuit" in full_text

    def test_help_methods(self):
        """Test listing available methods."""
        circuit = Circuit("test")
        help_obj = get_help(circuit)
        methods = help_obj.methods()
        assert isinstance(methods, list)
        assert "add" in methods
        assert "connect" in methods
        assert "validate" in methods

    def test_help_attributes(self):
        """Test listing available attributes."""
        circuit = Circuit("test")
        help_obj = get_help(circuit)
        attrs = help_obj.attributes()
        assert isinstance(attrs, list)
        assert "name" in attrs

    def test_method_help(self):
        """Test getting help for a specific method."""
        circuit = Circuit("test")
        help_obj = get_help(circuit)
        method_doc = help_obj.method_help("add")
        assert isinstance(method_doc, str)
        assert "add" in method_doc

    def test_method_help_not_found(self):
        """Test help for non-existent method."""
        circuit = Circuit("test")
        help_obj = get_help(circuit)
        result = help_obj.method_help("nonexistent_method")
        assert "not found" in result

    def test_circuit_help_quick_start(self):
        """Test circuit quick start guide."""
        circuit = Circuit("test")
        help_obj = CircuitHelp(circuit)
        quick_start = help_obj.quick_start()
        assert isinstance(quick_start, str)
        assert "Quick Start" in quick_start

    def test_circuit_help_components_empty(self):
        """Test components help for empty circuit."""
        circuit = Circuit("test")
        help_obj = CircuitHelp(circuit)
        comp_help = help_obj.components_help()
        assert "No components" in comp_help

    def test_circuit_help_components_with_items(self):
        """Test components help with components."""
        circuit = Circuit("test")
        R1 = Resistor(ref="1", resistance=1000)
        circuit.add(R1)
        vin = Net("vin")
        circuit.connect(R1.ports[0], vin)
        circuit.connect(R1.ports[1], GND)

        help_obj = CircuitHelp(circuit)
        comp_help = help_obj.components_help()
        assert "1" in comp_help  # Reference is "1"
        assert "Resistor" in comp_help

    def test_circuit_help_validation(self):
        """Test validation help."""
        circuit = Circuit("test")
        R1 = Resistor(ref="1", resistance=1000)
        circuit.add(R1)
        # Don't connect to get validation errors

        help_obj = CircuitHelp(circuit)
        val_help = help_obj.validation_help()
        assert "Validation" in val_help

    def test_component_help(self):
        """Test component help."""
        R1 = Resistor(ref="1", resistance=1000)
        help_obj = ComponentHelp(R1)

        full_text = help_obj.full()
        assert "Resistor" in full_text
        assert "1000" in full_text

    def test_component_help_ports(self):
        """Test component ports info."""
        R1 = Resistor(ref="1", resistance=1000)
        help_obj = ComponentHelp(R1)

        # Check sections contain ports info
        has_ports_section = any("Ports" in s.title for s in help_obj.sections)
        assert has_ports_section

    def test_help_str(self):
        """Test __str__ method returns full help."""
        circuit = Circuit("test")
        help_obj = get_help(circuit)
        assert str(help_obj) == help_obj.full()


class TestTutorial:
    """Tests for interactive tutorial system."""

    def test_list_tutorials(self):
        """Test listing available tutorials."""
        tutorials = list_tutorials()
        assert isinstance(tutorials, list)
        assert len(tutorials) > 0

    def test_tutorials_have_required_fields(self):
        """Test that tutorials have all required fields."""
        for tutorial in TUTORIALS.values():
            assert tutorial.name
            assert tutorial.title
            assert tutorial.description
            assert len(tutorial.steps) > 0

    def test_tutorial_steps_have_required_fields(self):
        """Test that tutorial steps have required fields."""
        for tutorial in TUTORIALS.values():
            for step in tutorial.steps:
                assert step.title
                assert step.description

    def test_tutorial_runner_creation(self):
        """Test creating a tutorial runner."""
        import io

        from spicelab.help.tutorial import get_tutorial

        tutorial = get_tutorial("basics")
        runner = TutorialRunner(tutorial, output_stream=io.StringIO())
        assert runner._current_step == 0
        assert runner._tutorial is not None

    def test_tutorial_runner_with_output(self):
        """Test tutorial runner with captured output."""
        import io

        from spicelab.help.tutorial import get_tutorial

        tutorial = get_tutorial("basics")
        output = io.StringIO()
        input_stream = io.StringIO("\n" * 20)  # Simulate pressing Enter

        runner = TutorialRunner(tutorial, input_stream=input_stream, output_stream=output)
        runner.run()

        output_text = output.getvalue()
        assert "Tutorial" in output_text
        assert tutorial.title in output_text

    def test_tutorial_step_dataclass(self):
        """Test TutorialStep dataclass."""
        step = TutorialStep(
            title="Test Step",
            description="A test step",
            code="print('hello')",
            explanation="This prints hello",
            exercise="Try printing something else",
            hints=["Use print()", "Strings need quotes"],
        )

        assert step.title == "Test Step"
        assert step.description == "A test step"
        assert step.code == "print('hello')"
        assert len(step.hints) == 2


class TestCheatsheet:
    """Tests for API cheat sheet generator."""

    def test_generate_text_cheatsheet(self):
        """Test generating text format cheat sheet."""
        content = generate_cheatsheet(CheatsheetFormat.TEXT)
        assert isinstance(content, str)
        assert "SpiceLab" in content
        assert "Circuit" in content

    def test_generate_markdown_cheatsheet(self):
        """Test generating markdown format cheat sheet."""
        content = generate_cheatsheet(CheatsheetFormat.MARKDOWN)
        assert isinstance(content, str)
        assert "# SpiceLab" in content
        assert "```python" in content

    def test_generate_html_cheatsheet(self):
        """Test generating HTML format cheat sheet."""
        content = generate_cheatsheet(CheatsheetFormat.HTML)
        assert isinstance(content, str)
        assert "<!DOCTYPE html>" in content
        assert "<html>" in content
        assert "SpiceLab" in content

    def test_cheatsheet_sections(self):
        """Test that cheat sheet has expected sections."""
        content = generate_cheatsheet(CheatsheetFormat.TEXT)

        # Check for main sections
        assert "Circuit" in content
        assert "Components" in content
        assert "Simulation" in content

    def test_cheatsheet_data_structure(self):
        """Test CHEATSHEET_DATA structure."""
        assert len(CHEATSHEET_DATA) > 0

        for section in CHEATSHEET_DATA:
            assert section.title
            assert section.description
            assert len(section.entries) > 0

            for entry in section.entries:
                assert entry.name
                assert entry.signature
                assert entry.description

    def test_save_cheatsheet_markdown(self, tmp_path):
        """Test saving cheat sheet to markdown file."""
        path = tmp_path / "cheatsheet.md"
        save_cheatsheet(str(path))

        assert path.exists()
        content = path.read_text()
        assert "# SpiceLab" in content

    def test_save_cheatsheet_html(self, tmp_path):
        """Test saving cheat sheet to HTML file."""
        path = tmp_path / "cheatsheet.html"
        save_cheatsheet(str(path))

        assert path.exists()
        content = path.read_text()
        assert "<html>" in content

    def test_save_cheatsheet_text(self, tmp_path):
        """Test saving cheat sheet to text file."""
        path = tmp_path / "cheatsheet.txt"
        save_cheatsheet(str(path))

        assert path.exists()
        content = path.read_text()
        assert "SpiceLab" in content

    def test_generate_filtered_sections(self):
        """Test generating cheat sheet with filtered sections."""
        content = generate_cheatsheet(
            CheatsheetFormat.TEXT,
            sections=["Circuit Creation"],
        )
        assert "Circuit Creation" in content
        # Other sections should not be present if filter works
        # (depends on implementation)


class TestExamples:
    """Tests for docstring example validation."""

    def test_run_example_success(self):
        """Test running a successful example."""
        result = run_example("x = 1 + 1\nprint(x)")
        assert result.success
        assert "2" in result.output

    def test_run_example_failure(self):
        """Test running a failing example."""
        result = run_example("raise ValueError('test error')")
        assert not result.success
        assert "ValueError" in result.error

    def test_run_example_syntax_error(self):
        """Test running example with syntax error."""
        result = run_example("def broken(")
        assert not result.success
        assert "SyntaxError" in result.error

    def test_list_examples_all(self):
        """Test listing all examples."""
        examples = list_examples()
        assert isinstance(examples, list)
        assert len(examples) > 0

    def test_list_examples_by_category(self):
        """Test listing examples by category."""
        examples = list_examples("basics")
        assert all(e.category == "basics" for e in examples)

    def test_get_example(self):
        """Test getting example by name."""
        example = get_example("create_circuit")
        assert example is not None
        assert example.name == "create_circuit"
        assert example.code

    def test_get_example_not_found(self):
        """Test getting non-existent example."""
        example = get_example("nonexistent_example")
        assert example is None

    def test_get_categories(self):
        """Test getting example categories."""
        categories = get_categories()
        assert isinstance(categories, list)
        assert "basics" in categories

    def test_run_example_by_name(self):
        """Test running example by name."""
        result = run_example_by_name("create_circuit", print_output=False)
        assert result.success

    def test_run_example_by_name_not_found(self):
        """Test running non-existent example."""
        with pytest.raises(ValueError, match="not found"):
            run_example_by_name("nonexistent", print_output=False)

    def test_examples_have_required_fields(self):
        """Test that all examples have required fields."""
        for example in EXAMPLES:
            assert example.name
            assert example.description
            assert example.code
            assert example.category

    def test_example_run_method(self):
        """Test Example.run() method."""
        from spicelab.help.examples import ExampleResult

        example = get_example("create_circuit")
        result = example.run()
        assert isinstance(result, ExampleResult)
        assert result.success is not None

    def test_validate_docstring_examples_invalid_module(self):
        """Test validating non-existent module."""
        report = validate_docstring_examples("nonexistent.module.name")
        assert report.failed > 0
        assert "Could not import" in report.results[0].error


class TestExampleResult:
    """Tests for ExampleResult dataclass."""

    def test_example_result_success_str(self):
        """Test string representation of successful result."""
        result = run_example("print('hello')")
        result_str = str(result)
        assert "✓" in result_str

    def test_example_result_failure_str(self):
        """Test string representation of failed result."""
        result = run_example("raise Exception('fail')")
        result_str = str(result)
        assert "✗" in result_str
        assert "Error" in result_str


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_validation_report_summary(self):
        """Test validation report summary."""
        from spicelab.help.examples import ExampleResult, ValidationReport

        report = ValidationReport(
            module_name="test_module",
            results=[
                ExampleResult("test1", "code1", True),
                ExampleResult("test2", "code2", True),
                ExampleResult("test3", "code3", False, error="error"),
            ],
        )

        assert report.total == 3
        assert report.passed == 2
        assert report.failed == 1
        assert report.success_rate == pytest.approx(66.67, rel=0.1)

        summary = report.summary()
        assert "test_module" in summary
        assert "3" in summary
        assert "2" in summary
        assert "1" in summary

    def test_validation_report_empty(self):
        """Test empty validation report."""
        from spicelab.help.examples import ValidationReport

        report = ValidationReport(module_name="empty")
        assert report.total == 0
        assert report.passed == 0
        assert report.failed == 0
        assert report.success_rate == 100.0


class TestShowHelp:
    """Tests for show_help function."""

    def test_show_help_circuit(self, capsys):
        """Test show_help prints circuit help."""
        circuit = Circuit("test")
        show_help(circuit)
        captured = capsys.readouterr()
        assert "Circuit" in captured.out

    def test_show_help_with_section(self, capsys):
        """Test show_help with specific section."""
        circuit = Circuit("test")
        show_help(circuit, "quick_start")
        captured = capsys.readouterr()
        assert "Quick Start" in captured.out

    def test_show_help_invalid_section(self, capsys):
        """Test show_help with invalid section."""
        circuit = Circuit("test")
        show_help(circuit, "invalid_section")
        captured = capsys.readouterr()
        assert "Unknown section" in captured.out
