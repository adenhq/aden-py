#!/bin/bash
set -e

# Build and publish script for aden-py
# Usage: ./scripts/publish.sh [major|minor|patch] [--dry-run]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYPROJECT="$PROJECT_ROOT/pyproject.toml"
INIT_FILE="$PROJECT_ROOT/src/aden/__init__.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
BUMP_TYPE=""
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        major|minor|patch)
            BUMP_TYPE="$arg"
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        --help|-h)
            echo "Usage: $0 [major|minor|patch] [--dry-run]"
            echo ""
            echo "Arguments:"
            echo "  major      Bump major version (x.0.0)"
            echo "  minor      Bump minor version (0.x.0)"
            echo "  patch      Bump patch version (0.0.x)"
            echo "  --dry-run  Show what would happen without making changes"
            echo ""
            echo "Environment:"
            echo "  PYPI_TOKEN  PyPI API token for publishing"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown argument: $arg${NC}"
            exit 1
            ;;
    esac
done

# Get current version from pyproject.toml
get_current_version() {
    grep -E '^version = "' "$PYPROJECT" | sed 's/version = "\(.*\)"/\1/'
}

# Bump version based on semver
bump_version() {
    local version="$1"
    local bump_type="$2"

    IFS='.' read -r major minor patch <<< "$version"

    case $bump_type in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
    esac

    echo "${major}.${minor}.${patch}"
}

# Update version in pyproject.toml
update_pyproject_version() {
    local new_version="$1"
    sed -i "s/^version = \".*\"/version = \"$new_version\"/" "$PYPROJECT"
}

# Update version in __init__.py
update_init_version() {
    local new_version="$1"
    sed -i "s/__version__ = \".*\"/__version__ = \"$new_version\"/" "$INIT_FILE"
}

# Main script
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  aden-py Build & Publish Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get current version
CURRENT_VERSION=$(get_current_version)
echo -e "Current version: ${YELLOW}$CURRENT_VERSION${NC}"

# Calculate new version if bump type specified
if [ -n "$BUMP_TYPE" ]; then
    NEW_VERSION=$(bump_version "$CURRENT_VERSION" "$BUMP_TYPE")
    echo -e "New version:     ${GREEN}$NEW_VERSION${NC} ($BUMP_TYPE bump)"
else
    NEW_VERSION="$CURRENT_VERSION"
    echo -e "${YELLOW}No version bump specified, using current version${NC}"
fi

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo -e "${YELLOW}=== DRY RUN MODE ===${NC}"
    echo ""
    echo "Would perform the following actions:"
    echo "  1. Update version in pyproject.toml to $NEW_VERSION"
    echo "  2. Update version in src/aden/__init__.py to $NEW_VERSION"
    echo "  3. Clean previous builds"
    echo "  4. Build package with hatchling"
    echo "  5. Upload to PyPI using twine"
    echo ""
    echo -e "${YELLOW}Run without --dry-run to execute${NC}"
    exit 0
fi

# Check for PYPI_TOKEN
if [ -z "$PYPI_TOKEN" ]; then
    echo ""
    echo -e "${RED}Error: PYPI_TOKEN environment variable not set${NC}"
    echo "Please set your PyPI API token:"
    echo "  export PYPI_TOKEN=pypi-..."
    exit 1
fi

# Update versions if bumping
if [ -n "$BUMP_TYPE" ]; then
    echo ""
    echo -e "${BLUE}Updating version files...${NC}"
    update_pyproject_version "$NEW_VERSION"
    echo "  ✓ Updated pyproject.toml"
    update_init_version "$NEW_VERSION"
    echo "  ✓ Updated src/aden/__init__.py"
fi

# Clean previous builds
echo ""
echo -e "${BLUE}Cleaning previous builds...${NC}"
rm -rf "$PROJECT_ROOT/dist" "$PROJECT_ROOT/build" "$PROJECT_ROOT"/*.egg-info "$PROJECT_ROOT/src"/*.egg-info
echo "  ✓ Cleaned dist/, build/, and egg-info directories"

# Install build dependencies if needed
echo ""
echo -e "${BLUE}Checking build dependencies...${NC}"
pip install --quiet --upgrade build twine
echo "  ✓ Build dependencies ready"

# Build the package
echo ""
echo -e "${BLUE}Building package...${NC}"
cd "$PROJECT_ROOT"
python -m build
echo "  ✓ Package built successfully"

# Show built files
echo ""
echo -e "${BLUE}Built artifacts:${NC}"
ls -la "$PROJECT_ROOT/dist/"

# Upload to PyPI
echo ""
echo -e "${BLUE}Uploading to PyPI...${NC}"
python -m twine upload \
    --username __token__ \
    --password "$PYPI_TOKEN" \
    --non-interactive \
    "$PROJECT_ROOT/dist/"*

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Successfully published aden-py $NEW_VERSION${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Install with:"
echo "  pip install aden-py==$NEW_VERSION"
echo ""
echo "PyPI page:"
echo "  https://pypi.org/project/aden-py/$NEW_VERSION/"
