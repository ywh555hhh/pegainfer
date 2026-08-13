use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use clap::Parser;
use clap::ValueEnum;
use pegainfer_higgs_audio::codec_input::CodeRowsLayout as RuntimeCodeRowsLayout;
use pegainfer_higgs_audio::codec_input::codec_input_from_rows;
use pegainfer_higgs_audio::codec_input::load_code_rows_json;
use pegainfer_higgs_audio::codec_input::write_codec_input_json;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum CodeRowsLayout {
    /// Input JSON contains delayed rows, shape [L, 8].
    Delayed,
    /// Input JSON contains already de-delayed raw codec rows, shape [T, 8].
    Raw,
}

#[derive(Parser)]
#[command(
    about = "Decode Higgs Audio codebook ids into a wav through the temporary Python codec sidecar"
)]
struct Args {
    /// Original Higgs checkpoint directory containing bundled codec weights.
    #[arg(long)]
    model_dir: PathBuf,
    /// JSON array or object containing delayed_codes/raw_codes.
    #[arg(long)]
    codes_json: PathBuf,
    /// Layout of --codes-json.
    #[arg(long, value_enum, default_value_t = CodeRowsLayout::Delayed)]
    codes_layout: CodeRowsLayout,
    /// Output wav path.
    #[arg(long)]
    out_wav: PathBuf,
    /// Optional path to persist the sanitized codec input JSON.
    #[arg(long)]
    codec_input_out: Option<PathBuf>,
    /// Python executable with torch/transformers/torchaudio/safetensors installed.
    #[arg(long, default_value = "python3")]
    python: String,
    /// Torch device used by the codec sidecar.
    #[arg(long, default_value = "cuda:0")]
    device: String,
    /// Stop after writing sanitized codec input JSON; do not launch Python codec.
    #[arg(long)]
    prepare_only: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let layout: RuntimeCodeRowsLayout = args.codes_layout.into();
    let rows = load_code_rows_json(&args.codes_json, layout)?;
    let codec_rows = codec_input_from_rows(&rows, layout)?;
    let codec_input = match &args.codec_input_out {
        Some(path) => {
            write_codec_input_json(path, &codec_rows)?;
            path.clone()
        }
        None => {
            let path = temp_codec_input_path();
            write_codec_input_json(&path, &codec_rows)?;
            path
        }
    };

    if !args.prepare_only {
        run_sidecar(&args, &codec_input)?;
    }

    if args.codec_input_out.is_none() && !args.prepare_only {
        let _ = std::fs::remove_file(&codec_input);
    }

    println!("higgs vocode codes: ok");
    println!("  model_dir: {}", args.model_dir.display());
    println!("  codes_json: {}", args.codes_json.display());
    println!("  codes_layout: {:?}", args.codes_layout);
    println!("  codec_frames: {}", codec_rows.len());
    if args.prepare_only {
        println!("  prepare_only: true");
    } else {
        println!("  out_wav: {}", args.out_wav.display());
    }
    if let Some(path) = args.codec_input_out {
        println!("  codec_input: {}", path.display());
    }
    Ok(())
}

fn run_sidecar(args: &Args, codec_input: &Path) -> Result<()> {
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("Higgs crate should live under the workspace root")?;
    let sidecar = repo_root.join("tools/higgs/vocode_higgs_codes.py");
    let mut command = Command::new(&args.python);
    command
        .arg(&sidecar)
        .arg("--model-dir")
        .arg(&args.model_dir)
        .arg("--codec-input-json")
        .arg(codec_input)
        .arg("--out-wav")
        .arg(&args.out_wav)
        .arg("--device")
        .arg(&args.device);
    let status = command
        .status()
        .with_context(|| format!("launch Python sidecar {}", sidecar.display()))?;
    if !status.success() {
        bail!("Higgs codec sidecar failed with status {status}");
    }
    Ok(())
}

fn temp_codec_input_path() -> PathBuf {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir().join(format!("higgs-codec-input-{pid}-{nanos}.json"))
}

impl From<CodeRowsLayout> for RuntimeCodeRowsLayout {
    fn from(value: CodeRowsLayout) -> Self {
        match value {
            CodeRowsLayout::Delayed => Self::Delayed,
            CodeRowsLayout::Raw => Self::Raw,
        }
    }
}
