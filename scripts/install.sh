#!/bin/sh
set -eu

arktor_version=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --version)
            if [ "$#" -lt 2 ] || [ -z "$2" ]; then
                printf '%s\n' "Usage: install.sh [--version X.Y.Z]" >&2
                exit 2
            fi
            arktor_version="$2"
            shift 2
            ;;
        -h|--help)
            printf '%s\n' "Usage: install.sh [--version X.Y.Z]"
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

if command -v uv >/dev/null 2>&1; then
    arktor_uv="$(command -v uv)"
else
    arktor_uv_dir="${UV_INSTALL_DIR:-${XDG_BIN_HOME:-$HOME/.local/bin}}"
    arktor_uv="$arktor_uv_dir/uv"
    if [ ! -x "$arktor_uv" ]; then
        if ! command -v curl >/dev/null 2>&1; then
            printf '%s\n' "curl is required to install uv." >&2
            exit 1
        fi
        printf 'uv was not found. Installing uv to %s...\n' "$arktor_uv_dir"
        arktor_installer="$(mktemp)"
        trap 'rm -f "$arktor_installer"' 0
        curl -fsSL https://astral.sh/uv/install.sh -o "$arktor_installer"
        UV_NO_MODIFY_PATH=1 UV_INSTALL_DIR="$arktor_uv_dir" sh "$arktor_installer"
    fi
fi

arktor_spec="arktor[cli,app,sandbox]"
if [ -n "$arktor_version" ]; then
    arktor_spec="$arktor_spec==$arktor_version"
fi

printf '%s\n' "Installing or upgrading Arktor..."
"$arktor_uv" tool install --upgrade --python 3.11 "$arktor_spec"
arktor_bin_dir="$("$arktor_uv" tool dir --bin)"
"$arktor_bin_dir/arktor" --version
printf 'Arktor installed at: %s/arktor\n' "$arktor_bin_dir"
printf '%s\n' "Configuring shell PATH..."
if "$arktor_uv" tool update-shell; then
    if [ "$(command -v arktor || true)" = "$arktor_bin_dir/arktor" ]; then
        printf '%s\n' "Run: arktor"
    else
        printf '%s\n' "Restart your terminal, then run: arktor"
    fi
else
    printf '%s\n' "PATH setup was not confirmed. See the message above." >&2
fi
