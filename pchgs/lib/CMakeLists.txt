CPMAddPackage(
        NAME fmt
        GIT_TAG 8.1.1
        GITHUB_REPOSITORY fmtlib/fmt
        OPTIONS "FMT_INSTALL YES" # create an installable target
)

CPMAddPackage(
        NAME nlohmann_json
        GITHUB_REPOSITORY nlohmann/json
        VERSION 3.9.1)

CPMAddPackage(
        NAME dynamic_bitset
        GITHUB_REPOSITORY pinam45/dynamic_bitset
        VERSION 1.2.1)
