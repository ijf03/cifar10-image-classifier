{
	description = "CPU-only Python environment for a CIFAR-10 classifier";

	inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

	outputs = { nixpkgs, ... }:
		let
			system = "x86_64-linux";
			pkgs = import nixpkgs {
				inherit system;
				config.cudaSupport = false;
			};
			python = pkgs.python3.withPackages (ps: with ps; [
				torch
				torchvision
				onnx
				onnxscript
			]);
		in
		{
			devShells.${system}.default = pkgs.mkShell {
				packages = [ 
					python
					pkgs.nodejs_22
				];
			};
		};
}
