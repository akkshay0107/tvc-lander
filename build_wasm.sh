#!/usr/bin/env bash

set -e

HELP_STRING=$(
	cat <<-END
		usage: build_wasm.sh PROJECT_NAME [--release]
		Build script for combining a Macroquad project with wasm-bindgen,
		allowing integration with the greater wasm-ecosystem.
		example: ./build_wasm.sh flappy-bird
		  This'll go through the following steps:
			    1. Build as target 'wasm32-unknown-unknown'.
			    2. Create the directory 'dist' if it doesn't already exist.
			    3. Run wasm-bindgen with output into the 'dist' directory.
		            - If the '--release' flag is provided, the build will be optimized for release.
			    4. Apply patches to the output js file (detailed here: https://github.com/not-fl3/macroquad/issues/212#issuecomment-835276147).
			    5. Generate coresponding 'index.html' file.
			Author: Tom Solberg <me@sbg.dev>
			Edit: Nik codes <nik.code.things@gmail.com>
			Edit: Nobbele <realnobbele@gmail.com>
			Edit: profan <robinhubner@gmail.com>
			Edit: Nik codes <nik.code.things@gmail.com>
			Edit: Akkshay Sundara Rajan <akkshaysr0107@gmail.com>
			Version: 0.5
	END
)

die() {
	echo >&2 "$HELP_STRING"
	echo >&2
	echo >&2 "Error: $*"
	exit 1
}

# Parse primary commands
while [[ $# -gt 0 ]]; do
	key="$1"
	case $key in
	--release)
		RELEASE=yes
		shift
		;;

	-h | --help)
		echo "$HELP_STRING"
		exit 0
		;;

	*)
		POSITIONAL+=("$1")
		shift
		;;
	esac
done

# Restore positionals
set -- "${POSITIONAL[@]}"
if [ $# -eq 0 ]; then
    die "missing PROJECT_NAME argument"
elif [ $# -gt 1 ]; then
    die "too many arguments provided"
fi

PROJECT_NAME=$1

BUILD_ID=$(date +%s)

HTML=$(
	cat <<-END
		<html lang="en">
		<head>
		    <meta charset="utf-8">
		    <title>${PROJECT_NAME}</title>
		<style>
		    html,
		    body {
		        margin: 0px;
		        padding: 0px;
		        width: 100%;
		        height: 100%;
		        overflow: hidden;
		        background-color: #050505;
		        display: flex;
		        justify-content: center;
		        align-items: center;
		        font-family: sans-serif;
		    }
		    canvas {
		        margin: 0px;
		        padding: 0px;
		        display: block;
		        width: 95vw;
		        height: auto;
		        max-height: 95vh;
		        max-width: calc(95vh * 16 / 9);
		        aspect-ratio: 16 / 9;
		        image-rendering: -webkit-optimize-contrast;
		        image-rendering: crisp-edges;
		        background-color: black;
		    }
		    #run-container {
		        display: flex;
		        justify-content: center;
		        align-items: center;
		        height: 100%;
		        width: 100%;
		        flex-direction: column;
		        text-align: center;
		        color: white;
		        background-color: #050505;
		        position: absolute;
		        top: 0;
		        left: 0;
		        z-index: 10;
		    }
		    #run-container h1 {
		        margin-top: 0;
		    }
		    #run-container p {
		        margin-bottom: 1.5em;
		        max-width: 600px;
		        padding: 0 20px;
		    }
		    #run-container button {
		        font-size: 1.2em;
		        padding: 10px 20px;
		        cursor: pointer;
		        background: #333;
		        color: white;
		        border: 1px solid #555;
		        border-radius: 4px;
		    }
		    #run-container button:hover {
		        background: #444;
		    }
		</style>
	</head>
	<body>
	    <canvas id="glcanvas" tabindex='1' hidden></canvas>
	    <script src="https://not-fl3.github.io/miniquad-samples/mq_js_bundle.js"></script>
	    <script type="module">
	        import init, { set_wasm } from "./${PROJECT_NAME}.js?v=${BUILD_ID}";
	        async function impl_run() {
	            let wbg = await init();
	            miniquad_add_plugin({
	                register_plugin: (a) => (a.wbg = wbg),
	                on_init: () => set_wasm(wasm_exports),
	                version: "0.0.1",
	                name: "wbg",
	            });
	            load("./${PROJECT_NAME}_bg.wasm?v=${BUILD_ID}");
	        }
	        window.run = function() {
	            document.getElementById("run-container").style.display = "none";
	            document.getElementById("glcanvas").removeAttribute("hidden");
	            document.getElementById("glcanvas").focus();
	            impl_run();
	        }
	    </script>
	    <div id="run-container">
	        <h1>tvc-lander</h1>
	        <p>
	            This is a simulation of a rocket landing using a Proximal Policy Optimization (PPO) agent for thrust vector control.
	            The agent will try to land the rocket safely on the landing pad.
	        </p>
	        <p>
	            Click and drag the rocket to move it to a new position and watch the agent attempt to recover and land.
	        </p>
	        <button onclick="run()">Run Sim</button>
	    </div>
	</body>
		</html>
	END
)

TARGET_DIR="target/wasm32-unknown-unknown"

# edited to specify binary in my code
if [ -n "$RELEASE" ]; then
	cargo build --bin $PROJECT_NAME --release --target wasm32-unknown-unknown
	TARGET_DIR="$TARGET_DIR/release"
else
	cargo build --bin $PROJECT_NAME --target wasm32-unknown-unknown
	TARGET_DIR="$TARGET_DIR/debug"
fi

# Generate bindgen outputs
mkdir -p dist
wasm-bindgen $TARGET_DIR/"$PROJECT_NAME".wasm --out-dir dist --target web --no-typescript

# Shim to tie the thing together
sed -i "s/import \* as __wbg_star0 from 'env';//" dist/"$PROJECT_NAME".js
sed -i "s/let wasm;/let wasm; export const set_wasm = (w) => wasm = w;/" dist/"$PROJECT_NAME".js
sed -i "s/imports\['env'\] = __wbg_star0;/return imports.wbg\;/" dist/"$PROJECT_NAME".js
sed -i "s/const imports = __wbg_get_imports();/return __wbg_get_imports();/" dist/"$PROJECT_NAME".js

# Create index from the HTML variable
echo "$HTML" >dist/index.html
