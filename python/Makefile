.VENV_DIR=.venv
.VENV_BIN=$(.VENV_DIR)/bin
.VENV_REQUIREMENTS=$(.VENV_DIR)/requirements.txt.touch
.VENV_REQUIREMENTS_DEV=$(.VENV_DIR)/requirements-dev.txt.touch

python=python3

define HELP_BODY
Please use 'make [target]'.

TARGETS

  setup [python=<path>]       Set up virtualenv using the Python interpreter at <path>, defaults to $(python)

  dist                        Build wheel and source distributions

  format [check=1]            Apply formatters; use with 'check=1' to check instead of modify

  lint                        Apply linters

  test [args=<pytest args>]   Run tests; use with 'args=<pytest args>' to pass test arguments

  clean                       Remove all build artifacts

endef


.PHONY: help
help:
	@: $(info $(HELP_BODY))


.PHONY: setup
setup: $(.VENV_DIR)/touch


.PHONY: format
format: $(.VENV_REQUIREMENTS) $(.VENV_REQUIREMENTS_DEV)
ifdef check
	$(.VENV_BIN)/black -l 100 --check jet tests
	$(.VENV_BIN)/isort --profile black --check-only jet tests
else
	$(.VENV_BIN)/black -l 100 jet tests
	$(.VENV_BIN)/isort --profile black jet tests
endif


.PHONY: lint
lint: $(.VENV_REQUIREMENTS) $(.VENV_REQUIREMENTS_DEV)
	$(.VENV_BIN)/pylint jet tests


.PHONY: dist
dist: $(.VENV_REQUIREMENTS) $(.VENV_REQUIREMENTS_DEV)
	/bin/sh -c '. $(.VENV_BIN)/activate && cd .. && python setup.py -q sdist bdist_wheel'
	@# Delete the .egg-info artifact to prevent pip from assuming the package was installed.
	@# See https://github.com/pypa/pip/issues/6558 for more details.
	rm -rf quantum_jet.egg-info


.PHONY: test
test: dist
	$(.VENV_BIN)/pip install -q $(wildcard ../dist/*.whl) --upgrade
	@# The -I flag ensures the installed `jet` package is imported instead of the local one.
	$(.VENV_BIN)/python -I -m pytest ./tests -W ignore::UserWarning $(args)


.PHONY: clean
clean:
	rm -rf $(.VENV_DIR) quantum_jet.egg-info
	rm -rf ../.pytest_cache ../dist ../build/bdist.* ../build/lib.* ../build/temp.*


$(.VENV_REQUIREMENTS): $(.VENV_DIR)/touch requirements.txt
	$(.VENV_DIR)/bin/pip install -q -r requirements.txt
	@touch $@


$(.VENV_REQUIREMENTS_DEV): $(.VENV_DIR)/touch requirements-dev.txt
	$(.VENV_DIR)/bin/pip install -q -r requirements-dev.txt
	@touch $@


$(.VENV_DIR)/touch:
	$(python) -m venv ${.VENV_DIR}
	@touch $@
