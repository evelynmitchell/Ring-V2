#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
PATCH_FILE="${SCRIPT_DIR}/te.patch"
PACKAGE_DIR=$(python3 -c "import transformer_engine; print(transformer_engine.__path__[0])")

# Check if the patch file exists
if [[ ! -f "$PATCH_FILE" ]]; then
	  echo "Error: Patch file '$PATCH_FILE' not found."
	  exit 1
fi

# Check if the package directory exists
if [[ ! -d "$PACKAGE_DIR" ]]; then
	echo "Error: Package directory '$PACKAGE_DIR' not found."
	exit 1
fi

INSTALLED_VERSION=$(python3 -c "import pkg_resources; print(pkg_resources.get_distribution('transformer_engine').version)" 2>/dev/null)

TARGET_VERSION="2.6.0.post1"

if [ "$INSTALLED_VERSION" == "$TARGET_VERSION" ]; then
    echo patching transformer_engine in directory "$PACKAGE_DIR" with "$PATCH_FILE"
    patch -p1 -d "$PACKAGE_DIR" < "$PATCH_FILE"
else
    echo The installed version for transformer_engine is "$INSTALLED_VERSION", and patch only applies to "$TARGET_VERSION". Patching skipped.
fi


