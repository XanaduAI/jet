.VENV_DIR=.venv
.VENV_BIN=$(.VENV_DIR)/bin

define HELP_BODY
Please use 'make [target]'.

TARGETS

  setup                       setup virtualenv
  
  test                        Test bindings

  format [check=1]            Apply formatters; use with 'check=1' to check instead of modify

  clean                       Remove all build artifacts

endef

.PHONY: help
help:
	@: $(info $(HELP_BODY))

setup: $(.VENV_DIR)/requirements_test.txt.touch

.PHONY: format
format:
ifdef check
	$(.VENV_BIN)/black --check tests && $(.VENV_BIN)/isort --profile black --check-only tests
else
	$(.VENV_BIN)/black tests && $(.VENV_BIN)/isort --profile black tests
endif

.PHONY: test
test: $(.VENV_DIR)/requirements_test.txt.touch
	$(.VENV_BIN)/python3 -m pytest

.PHONY: clean
	rm -rf $(.VENV_DIR)

$(.VENV_DIR)/requirements_test.txt.touch: $(.VENV_DIR)/touch requirements_test.txt
	$(.VENV_DIR)/bin/pip install -r requirements_test.txt
	touch $@

$(.VENV_DIR)/touch:
	python3 -m venv $(.VENV_DIR)
	touch $@
