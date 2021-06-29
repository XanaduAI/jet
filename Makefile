.DEFAULT_PREFIX=/usr/local
.TEST_BUILD_DIR=./build
.VENV_DIR=./.venv

prefix=$(.DEFAULT_PREFIX)

define HELP_BODY
Please use 'make [target]'.

TARGETS

  install [prefix=<path>]     Install Jet headers to <path>/include, defaults to $(.DEFAULT_PREFIX)

  uninstall [prefix=<path>]   Remove Jet headers from <path>/include, defaults to $(.DEFAULT_PREFIX)

  test                        Build and run C++ tests (requires Cmake)

  docs                        Build docs (requires Doxygen, Pandoc and pip)

  format [check=1]            Apply C++ formatter; use with 'check=1' to check instead of modify (requires clang-format)

  clean                       Remove all build artifacts

endef

.PHONY: help
help:
	@: $(info $(HELP_BODY))


.PHONY: format
format:
ifdef check
	./bin/format --check include python/src test
else
	./bin/format include python/src test
endif


.PHONY: install
install: ./include/Jet.hpp ./include/jet
	mkdir -p $(prefix)/include
	cp -r $^ $(prefix)/include/


.PHONY: docs
uninstall:
	rm -r $(prefix)/include/jet $(prefix)/include/Jet.hpp


.PHONY: test
test: $(.TEST_BUILD_DIR)
	cd $(.TEST_BUILD_DIR) && $(MAKE)
	$(.TEST_BUILD_DIR)/test/runner


.PHONY: docs
docs: $(.VENV_DIR)
	. $(.VENV_DIR)/bin/activate; cd ./docs && $(MAKE) html


.PHONY: clean
clean:
	rm -rf ./docs/_build ./docs/api ./docs/code/api ./docs/doxyoutput
	rm -rf $(.TEST_BUILD_DIR)
	rm -rf $(.VENV_DIR)


$(.VENV_DIR):
	python3 -m venv $@
	$@/bin/pip install wheel
	$@/bin/pip install -r docs/requirements.txt


$(.TEST_BUILD_DIR):
	mkdir -p $@
	cd $@ && cmake -DBUILD_TESTS=ON ../
