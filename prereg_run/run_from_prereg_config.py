
import sys
import yaml
import subprocess
from pathlib import Path

def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    base_out = Path("prereg_runs")
    base_out.mkdir(exist_ok=True)

    architectures = cfg["factors"]["architecture"]
    decode = cfg["decode_controls"]
    seed = cfg["seed"]
    selection = cfg["selection_policy"]

    commands = []

    for arch, strategies in architectures.items():
        for strat in strategies:
            run_name = f"{arch}_{strat}"
            out_dir = base_out / run_name
            out_dir.mkdir(parents=True, exist_ok=True)

            if arch == "google":
                cmd = [
                    "python", "pipeline/translate_with_annotation.py",
                    "--google",
                    "--objective", "seq",
                    "--in", "data/ENArticles_PLTranslation.csv", "data/synth_PL_55.csv",
                    "--out", str(out_dir),
                    "--seed", str(seed),
                    "--annotate"
                ]
            else:
                cmd = [
                    "python", "pipeline/translate_with_annotation.py",
                    "--architecture", arch,
                    "--objective", strat,
                    "--in", "data/ENArticles_PLTranslation.csv", "data/synth_PL_55.csv",
                    "--out", str(out_dir),
                    "--beam", str(decode["beam"]),
                    "--top-p", str(decode["top_p"]),
                    "--temperature", str(decode["temperature"]),
                    "--max-new-tokens", str(decode["max_new_tokens"]),
                    "--seed", str(seed),
                    "--annotate"
                ]

                if selection["rt_as_gold"]:
                    cmd += ["--ft-gold-from"] + selection["metrics"]

            commands.append(cmd)

    # Save manifest
    manifest = base_out / "run_manifest.txt"
    with open(manifest, "w") as f:
        for c in commands:
            f.write(" ".join(c) + "\n")

    print(f"Generated {len(commands)} preregistered runs.")
    print(f"Manifest written to {manifest}")
    print("Starting execution...\n")

    for c in commands:
        print("RUN:", " ".join(c))
        subprocess.run(c, check=True)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_from_prereg_config.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
