.VENV_DIR=.venv
.VENV_BIN=$(.VENV_DIR)/bin

python=python3

define HELP_BODY
Please use 'make [target]'.

TARGETS

  setup [python=<path>]       Set up virtualenv using the Python interpreter at <path>, defaults to $(python)

  build                       Create source distribution and wheel

  format [check=1]            Apply formatters; use with 'check=1' to check instead of modify

  test [args=<pytest args>]   Run tests; use with 'args=<pytest args>' to pass test arguments

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
	$(.VENV_BIN)/black -l 100 --check jet tests
	$(.VENV_BIN)/isort --profile black --check-only jet tests
else
	$(.VENV_BIN)/black -l 100 jet tests
	$(.VENV_BIN)/isort --profile black jet tests
endif


.PHONY: build
build: $(.VENV_DIR)/requirements.txt.touch
	$(.VENV_BIN)/python -m build
	@# Delete the .egg-info artifact to prevent pip from assuming the package was installed.
	@# See https://github.com/pypa/pip/issues/6558 for more details.
	rm -rf quantum_jet.egg-info


.PHONY: test
test: build
	$(.VENV_BIN)/pip install $(wildcard dist/*.whl) --upgrade
	@# The -I flag ensures the installed `jet` package is imported instead of the local one.
	$(.VENV_BIN)/python -I -m pytest ./tests $(args)


.PHONY: clean
clean:
	rm -rf $(.VENV_DIR) build dist


$(.VENV_DIR)/requirements.txt.touch: $(.VENV_DIR)/touch requirements.txt
	$(.VENV_DIR)/bin/pip install wheel
	$(.VENV_DIR)/bin/pip install -r requirements.txt
	@touch $@


$(.VENV_DIR)/touch:
	$(python) -m venv ${.VENV_DIR}
	@touch $@
