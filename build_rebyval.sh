# Build python wheel package
build_wheel_package() {
  python3 setup.py develop
}

main() {
  build_wheel_package
}

main "$@"