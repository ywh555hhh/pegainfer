use anyhow::Context;
use anyhow::Result;
use anyhow::bail;

use crate::one_step_golden::CODEBOOK_VOCAB_SIZE;
use crate::one_step_golden::NUM_CODEBOOKS;

pub const BOC_ID: u32 = 1024;
pub const EOC_ID: u32 = 1025;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelayPatternState {
    num_codebooks: usize,
    delay_count: usize,
    eoc_countdown: Option<usize>,
    generation_done: bool,
    last_codes: Option<Vec<u32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelayPatternStep {
    pub codes: Option<Vec<u32>>,
    pub generation_done: bool,
    pub delay_count: usize,
    pub eoc_countdown: Option<usize>,
}

impl DelayPatternState {
    pub fn new(num_codebooks: usize) -> Result<Self> {
        if num_codebooks == 0 {
            bail!("Higgs delay pattern requires at least one codebook");
        }
        Ok(Self {
            num_codebooks,
            delay_count: 0,
            eoc_countdown: None,
            generation_done: false,
            last_codes: None,
        })
    }

    pub fn new_higgs_v3() -> Self {
        Self::new(NUM_CODEBOOKS).expect("pinned Higgs v3 codebook count is non-zero")
    }

    pub fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    pub fn delay_count(&self) -> usize {
        self.delay_count
    }

    pub fn eoc_countdown(&self) -> Option<usize> {
        self.eoc_countdown
    }

    pub fn generation_done(&self) -> bool {
        self.generation_done
    }

    pub fn last_codes(&self) -> Option<&[u32]> {
        self.last_codes.as_deref()
    }

    pub fn step_from_sampled_codes(&mut self, sampled_codes: &[u32]) -> Result<DelayPatternStep> {
        if sampled_codes.len() != self.num_codebooks {
            bail!(
                "sampled code row has {} codebooks, expected {}",
                sampled_codes.len(),
                self.num_codebooks
            );
        }
        for &code in sampled_codes {
            if code as usize >= CODEBOOK_VOCAB_SIZE {
                bail!(
                    "sampled code id {code} is outside Higgs codebook vocab size {}",
                    CODEBOOK_VOCAB_SIZE
                );
            }
        }

        if self.generation_done {
            return Ok(self.snapshot(None));
        }

        let mut codes = sampled_codes.to_vec();
        if self.delay_count < self.num_codebooks {
            let next_cb = self.delay_count + 1;
            if next_cb < self.num_codebooks {
                codes[next_cb..].fill(BOC_ID);
            }
            self.delay_count += 1;
        } else if let Some(countdown) = self.eoc_countdown {
            let next = countdown.saturating_sub(1);
            self.eoc_countdown = Some(next);
            if next == 0 {
                self.generation_done = true;
            }
        } else if codes[0] == EOC_ID {
            if self.num_codebooks <= 2 {
                self.generation_done = true;
            } else {
                self.eoc_countdown = Some(self.num_codebooks - 2);
            }
        }

        if !self.generation_done {
            self.last_codes = Some(codes.clone());
        }

        Ok(self.snapshot(Some(codes)))
    }

    fn snapshot(&self, codes: Option<Vec<u32>>) -> DelayPatternStep {
        DelayPatternStep {
            codes,
            generation_done: self.generation_done,
            delay_count: self.delay_count,
            eoc_countdown: self.eoc_countdown,
        }
    }
}

pub fn apply_delay_pattern(raw_codes: &[Vec<u32>], num_codebooks: usize) -> Result<Vec<Vec<u32>>> {
    if num_codebooks == 0 {
        bail!("Higgs delay pattern requires at least one codebook");
    }
    for (row_idx, row) in raw_codes.iter().enumerate() {
        if row.len() != num_codebooks {
            bail!(
                "raw code row {row_idx} has {} codebooks, expected {num_codebooks}",
                row.len()
            );
        }
        validate_real_code_row(row)
            .with_context(|| format!("raw code row {row_idx} contains invalid code"))?;
    }

    let rows = raw_codes.len() + num_codebooks - 1;
    let mut delayed = vec![vec![EOC_ID; num_codebooks]; rows];
    for codebook in 0..num_codebooks {
        for row in delayed.iter_mut().take(codebook) {
            row[codebook] = BOC_ID;
        }
        for (raw_idx, raw_row) in raw_codes.iter().enumerate() {
            delayed[codebook + raw_idx][codebook] = raw_row[codebook];
        }
    }
    Ok(delayed)
}

pub fn reverse_delay_pattern(delayed: &[Vec<u32>], allow_short: bool) -> Result<Vec<Vec<u32>>> {
    let Some(first_row) = delayed.first() else {
        if allow_short {
            return Ok(Vec::new());
        }
        bail!("delayed code matrix is empty");
    };
    let num_codebooks = first_row.len();
    if num_codebooks == 0 {
        bail!("delayed code matrix must have at least one codebook");
    }
    for (row_idx, row) in delayed.iter().enumerate() {
        if row.len() != num_codebooks {
            bail!(
                "delayed code row {row_idx} has {} codebooks, expected {num_codebooks}",
                row.len()
            );
        }
    }

    let Some(rows) = delayed.len().checked_sub(num_codebooks - 1) else {
        if allow_short {
            return Ok(Vec::new());
        }
        bail!(
            "delayed has L={}, N={num_codebooks}; need L >= N so at least one data row can be recovered",
            delayed.len()
        );
    };
    if rows == 0 {
        if allow_short {
            return Ok(Vec::new());
        }
        bail!(
            "delayed has L={}, N={num_codebooks}; need L >= N so at least one data row can be recovered",
            delayed.len()
        );
    }

    let mut raw = vec![vec![0; num_codebooks]; rows];
    for codebook in 0..num_codebooks {
        for raw_idx in 0..rows {
            raw[raw_idx][codebook] = delayed[codebook + raw_idx][codebook];
        }
    }
    Ok(raw)
}

pub fn delay_pattern_action_mask(delayed: &[Vec<u32>]) -> Result<Vec<Vec<bool>>> {
    let Some(first_row) = delayed.first() else {
        return Ok(Vec::new());
    };
    let num_codebooks = first_row.len();
    if num_codebooks == 0 {
        bail!("delayed code matrix must have at least one codebook");
    }
    for (row_idx, row) in delayed.iter().enumerate() {
        if row.len() != num_codebooks {
            bail!(
                "delayed code row {row_idx} has {} codebooks, expected {num_codebooks}",
                row.len()
            );
        }
    }

    let raw_rows = delayed
        .iter()
        .position(|row| row[0] == EOC_ID)
        .unwrap_or(delayed.len());
    let mut mask = vec![vec![false; num_codebooks]; delayed.len()];
    for (row_idx, row) in mask.iter_mut().enumerate() {
        for (codebook, cell) in row.iter_mut().enumerate() {
            *cell = codebook <= row_idx && row_idx < codebook + raw_rows;
        }
    }
    Ok(mask)
}

fn validate_real_code_row(row: &[u32]) -> Result<()> {
    for &code in row {
        if code >= BOC_ID {
            bail!("real audio code {code} overlaps Higgs BOC/EOC sentinel range");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(base: u32, n: usize) -> Vec<u32> {
        (0..n).map(|idx| base + idx as u32).collect()
    }

    #[test]
    fn step_forces_boc_during_ramp_up_window() {
        let mut state = DelayPatternState::new(4).unwrap();

        let first = state.step_from_sampled_codes(&row(10, 4)).unwrap();
        assert_eq!(first.codes.unwrap(), vec![10, BOC_ID, BOC_ID, BOC_ID]);
        assert_eq!(state.delay_count(), 1);
        assert_eq!(
            state.last_codes(),
            Some([10, BOC_ID, BOC_ID, BOC_ID].as_slice())
        );

        let second = state.step_from_sampled_codes(&row(20, 4)).unwrap();
        assert_eq!(second.codes.unwrap(), vec![20, 21, BOC_ID, BOC_ID]);
        assert_eq!(state.delay_count(), 2);

        let third = state.step_from_sampled_codes(&row(30, 4)).unwrap();
        assert_eq!(third.codes.unwrap(), vec![30, 31, 32, BOC_ID]);
        assert_eq!(state.delay_count(), 3);

        let fourth = state.step_from_sampled_codes(&row(40, 4)).unwrap();
        assert_eq!(fourth.codes.unwrap(), vec![40, 41, 42, 43]);
        assert_eq!(state.delay_count(), 4);
        assert_eq!(state.eoc_countdown(), None);
        assert!(!state.generation_done());
    }

    #[test]
    fn shortest_two_codebook_sequence_finishes_on_cb0_eoc() {
        let mut state = DelayPatternState::new(2).unwrap();

        assert_eq!(
            state
                .step_from_sampled_codes(&[1, 2])
                .unwrap()
                .codes
                .unwrap(),
            vec![1, BOC_ID]
        );
        assert_eq!(
            state
                .step_from_sampled_codes(&[3, 4])
                .unwrap()
                .codes
                .unwrap(),
            vec![3, 4]
        );
        let done = state.step_from_sampled_codes(&[EOC_ID, 7]).unwrap();
        assert_eq!(done.codes.unwrap(), vec![EOC_ID, 7]);
        assert!(done.generation_done);
        assert_eq!(state.last_codes(), Some([3, 4].as_slice()));

        let after_done = state.step_from_sampled_codes(&[8, 9]).unwrap();
        assert_eq!(after_done.codes, None);
        assert!(after_done.generation_done);
        assert_eq!(state.last_codes(), Some([3, 4].as_slice()));
    }

    #[test]
    fn cb0_eoc_after_delay_starts_winddown_for_later_codebooks() {
        let mut state = DelayPatternState::new(4).unwrap();
        for base in [10, 20, 30, 40] {
            state.step_from_sampled_codes(&row(base, 4)).unwrap();
        }

        let eoc = state
            .step_from_sampled_codes(&[EOC_ID, 51, 52, 53])
            .unwrap();
        assert_eq!(eoc.codes.unwrap(), vec![EOC_ID, 51, 52, 53]);
        assert!(!eoc.generation_done);
        assert_eq!(state.eoc_countdown(), Some(2));

        let winddown_1 = state.step_from_sampled_codes(&row(60, 4)).unwrap();
        assert_eq!(winddown_1.eoc_countdown, Some(1));
        assert!(!winddown_1.generation_done);
        assert_eq!(state.last_codes(), Some([60, 61, 62, 63].as_slice()));

        let winddown_2 = state.step_from_sampled_codes(&row(70, 4)).unwrap();
        assert_eq!(winddown_2.eoc_countdown, Some(0));
        assert!(winddown_2.generation_done);
        assert_eq!(state.last_codes(), Some([60, 61, 62, 63].as_slice()));
    }

    #[test]
    fn eoc_during_ramp_up_is_ignored_until_all_codebooks_are_active() {
        let mut state = DelayPatternState::new(4).unwrap();

        let first = state
            .step_from_sampled_codes(&[EOC_ID, 11, 12, 13])
            .unwrap();
        assert_eq!(first.codes.unwrap(), vec![EOC_ID, BOC_ID, BOC_ID, BOC_ID]);
        assert_eq!(state.eoc_countdown(), None);
        assert!(!state.generation_done());

        for base in [20, 30, 40] {
            state.step_from_sampled_codes(&row(base, 4)).unwrap();
        }
        assert_eq!(state.delay_count(), 4);
        assert_eq!(state.eoc_countdown(), None);
        assert!(!state.generation_done());
    }

    #[test]
    fn apply_and_reverse_delay_pattern_realigns_rows() {
        let raw = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let delayed = apply_delay_pattern(&raw, 3).unwrap();

        assert_eq!(
            delayed,
            vec![
                vec![1, BOC_ID, BOC_ID],
                vec![4, 2, BOC_ID],
                vec![7, 5, 3],
                vec![EOC_ID, 8, 6],
                vec![EOC_ID, EOC_ID, 9],
            ]
        );
        assert_eq!(reverse_delay_pattern(&delayed, false).unwrap(), raw);
    }

    #[test]
    fn action_mask_selects_real_audio_parallelogram() {
        let raw = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let delayed = apply_delay_pattern(&raw, 3).unwrap();
        let mask = delay_pattern_action_mask(&delayed).unwrap();

        assert_eq!(
            mask,
            vec![
                vec![true, false, false],
                vec![true, true, false],
                vec![false, true, true],
                vec![false, false, true],
            ]
        );
    }

    #[test]
    fn reverse_delay_pattern_can_accept_short_windows_for_streaming() {
        let short = vec![vec![1, BOC_ID, BOC_ID], vec![2, 3, BOC_ID]];
        assert_eq!(
            reverse_delay_pattern(&short, true).unwrap(),
            Vec::<Vec<u32>>::new()
        );
        assert!(reverse_delay_pattern(&short, false).is_err());
    }

    #[test]
    fn rejects_wrong_shapes_and_sentinel_as_real_audio() {
        let err = DelayPatternState::new(3)
            .unwrap()
            .step_from_sampled_codes(&[1, 2])
            .unwrap_err()
            .to_string();
        assert!(err.contains("expected 3"));

        let err = apply_delay_pattern(&[vec![1, BOC_ID]], 2)
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid code"));
    }
}
