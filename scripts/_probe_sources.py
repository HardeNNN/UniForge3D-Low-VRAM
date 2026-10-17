"""Probe every declared download URL/size (dev helper)."""
import sys

sys.path.insert(0, "scripts")
import fetch  # noqa: E402

for url, dest in fetch.WHEEL_FILES:
    try:
        print("%9.1f MB  %s" % (fetch.head_size(url) / 1e6, dest.name))
    except Exception as error:
        print("   FAIL %s: %s" % (dest.name, error))

try:
    url = fetch.pypi_wheel_url("triton-windows", "3.4.0.post21")
    print("%9.1f MB  %s" % (fetch.head_size(url) / 1e6, url.rsplit("/", 1)[-1]))
except Exception as error:
    print("   FAIL triton:", error)

for url, dest, _ in fetch.EXTRA_ZIPS:
    try:
        print("%9.1f MB  %s" % (fetch.head_size(url) / 1e6, dest.name))
    except Exception as error:
        print("   FAIL %s: %s" % (dest.name, error))

for (repo, revision), paths in fetch.HUB_FILES.items():
    try:
        sizes = fetch.hub_sizes(repo, revision, paths)
        print("%9.2f GB  %s" % (sum(sizes.values()) / 1e9, repo))
    except Exception as error:
        print("   FAIL %s: %s" % (repo, error))
