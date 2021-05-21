.VENV_DIR=.venv
.VENV_BIN=$(.VENV_DIR)/bin

python=python3

define HELP_BODY
Please use 'make [target]'.

TARGETS

  setup [python=<path>]       Set up virtualenv using the Python interpreter at <path>, defaults to $(python)

  test [args=<pytest args>]   Run tests; use with 'args=<pytest args>' to pass test arguments

  format [check=1]            Apply formatters; use with 'check=1' to check instead of modify

  clean                       Remove all build artifacts

endef


.PHONY: help
help:
	@: $(info $(HELP_BODY))


.PHONY: setup
setup: $(.VENV_DIR)/touch


.PHONY: format
format: $(.VENV_DIR)/requirements.txt.touch
ifdef check
	$(.VENV_BIN)/black --check tests
	$(.VENV_BIN)/isort --profile black --check-only tests
else
	$(.VENV_BIN)/black tests
	$(.VENV_BIN)/isort --profile black tests
endif


.PHONY: build
build: $(.VENV_DIR)/requirements.txt.touch
	$(.VENV_BIN)/python -m build -n ..
	rm -rf quantum_jet.egg-info


.PHONY: test
test: build
	$(.VENV_BIN)/pip install $(wildcard ../dist/*.whl)
	$(.VENV_BIN)/python -I -m pytest ./tests $(args)


.PHONY: clean
clean:
	rm -rf $(.VENV_DIR)


$(.VENV_DIR)/requirements.txt.touch: $(.VENV_DIR)/touch requirements.txt
	$(.VENV_DIR)/bin/pip install -r requirements.txt
	@touch $@


$(.VENV_DIR)/touch:
	$(python) -m venv ${.VENV_DIR}
	@touch $@
