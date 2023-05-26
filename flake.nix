{
  description = "Sample Direnv Flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:

    flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { 
        inherit system;
        config.allowUnfree = true;
      };

    in
    {
      devShells.default = pkgs.mkShell {
        packages = with pkgs; [
          valgrind
          perf-tools
          gperftools
        ];
        
        shellHook = ''
        '';
      };
    });
}
