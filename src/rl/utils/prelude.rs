

pub use super::statistics::*;


pub fn count_files(path: &str) -> std::io::Result<usize> {
    let mut count = 0;

    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;

        if metadata.is_file() {
            count += 1;
        }
    }

    Ok(count)
}
