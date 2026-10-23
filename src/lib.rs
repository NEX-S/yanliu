//! Local Mandarin dictionary decoding and revision-safe input sessions.
pub mod decoder;
pub mod dictionary;
#[cfg(feature = "ibus")]
pub mod ibus;
pub mod learning;
pub mod model;
pub mod session;
pub use decoder::{Candidate, CandidateSource, Decoder};
pub use dictionary::Dictionary;
pub use session::Session;

#[cfg(test)]
mod tests {
    use super::*;
    use learning::Learning;
    use model::Model;
    use std::{io::Cursor, path::Path, sync::Arc};
    fn decoder() -> Arc<Decoder> {
        Arc::new(Decoder::new(Dictionary::from_rime_reader(Cursor::new("name: test\n...\n你\tni\t100\n好\thao\t90\n世界\tshi jie\t80\n西\txi\t100\n安\tan\t90\n现\txian\t100\n西安\txi an\t120\n你好\tni hao\t120\n" )).unwrap()))
    }
    fn learning_session(decoder: Arc<Decoder>) -> Session {
        Session::new(decoder, Some(Arc::new(Learning::new(false))))
    }
    fn model() -> Model {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("data/models/sentence-model.safetensors");
        Model::load(&path, "cpu", None)
            .expect("download the pinned model with scripts/fetch_assets.py")
    }
    fn refine(model: &mut Model, decoder: &Decoder, session: &mut Session) {
        let mut snapshot = session.snapshot();
        let result = model
            .refine(&mut snapshot, decoder, || true)
            .unwrap()
            .unwrap();
        assert!(session.apply_candidates(
            snapshot.revision,
            snapshot.candidates,
            result.index,
            &result.order
        ));
    }
    #[test]
    fn composes_unlisted_sentence_and_completes_tail() {
        let d = decoder();
        assert!(d.decode("nihaoshijie").iter().any(|c| c.text == "你好世界"
            && c.consumed == 11
            && c.source == CandidateSource::Composed));
        assert!(
            d.decode("nihaoshij")
                .iter()
                .any(|c| c.text == "你好世界" && c.source == CandidateSource::Completion)
        );
    }
    #[test]
    fn separator_enforces_syllable_boundary() {
        let d = decoder();
        assert!(d.decode("xian").iter().any(|c| c.text == "现"));
        assert!(d.decode("xian").iter().any(|c| c.text == "西安"));
        let split = d.decode("xi'an");
        assert!(!split.iter().any(|c| c.text == "现"));
        assert!(split.iter().any(|c| c.text == "西安" && c.consumed == 5));
    }
    #[test]
    fn completion_budget_ignores_irrelevant_long_words() {
        let mut fixture = String::from("...\n你\tni\t100\n年\tnian\t90\n");
        for index in 0..256 {
            fixture.push_str(&format!("词{index}\tna a\t1000\n"));
        }
        let d = Decoder::new(Dictionary::from_rime_reader(Cursor::new(fixture)).unwrap());
        let candidates = d.decode("n");
        assert!(candidates.iter().any(|c| c.text == "你"));
        assert!(candidates.iter().any(|c| c.text == "年"));
    }

    #[test]
    fn trailing_separator_closes_the_syllable() {
        let d = Decoder::new(
            Dictionary::from_rime_reader(Cursor::new("...\n你\tni\t100\n年\tnian\t100\n")).unwrap(),
        );
        assert!(
            d.decode("ni")
                .iter()
                .any(|c| c.text == "年" && c.source == CandidateSource::Completion)
        );
        let closed = d.decode("ni'");
        assert!(closed.iter().any(|c| c.text == "你" && c.consumed == 3));
        assert!(!closed.iter().any(|c| c.text == "年"));
    }

    fn abbreviation_decoder() -> Decoder {
        Decoder::new(
            Dictionary::from_rime_reader(Cursor::new(
                "...\n女孩\tnv hai\t500\n你会\tni hui\t400\n你好\tni hao\t300\n拟好\tni hao\t10\n你\tni\t100\n好\thao\t90\n中国\tzhong guo\t200\n这个\tzhe ge\t500\n最高\tzui gao\t900\n上海\tshang hai\t80\n是\tshi\t100\n世界\tshi jie\t70\n",
            ))
            .unwrap(),
        )
    }
    #[test]
    fn abbreviation_recalls_initials_without_changing_full_pinyin() {
        let d = abbreviation_decoder();
        let nh = d.decode("nh");
        for text in ["女孩", "你会", "你好", "拟好"] {
            assert!(nh.iter().any(|c| c.text == text
                && c.consumed == 2
                && c.source == CandidateSource::ExactDictionary));
        }
        let at = |text: &str| nh.iter().position(|c| c.text == text).unwrap();
        assert!(at("女孩") < at("你会") && at("你会") < at("你好"));
        assert!(d.decode("nihao").iter().any(|c| c.text == "你好"
            && c.consumed == 5
            && c.source == CandidateSource::ExactDictionary));
        assert!(d.decode("nih").iter().any(|c| c.text == "你好"
            && c.consumed == 3
            && c.source == CandidateSource::Completion));
        assert!(d.decode("n").iter().all(|c| c.text != "你好"));
        assert!(d.decode("nhao").iter().any(|c| c.text == "你好"
            && c.consumed == 4
            && c.source == CandidateSource::ExactDictionary));
        assert!(d.decode("n'h").iter().any(|c| c.text == "你好"
            && c.consumed == 3
            && c.source == CandidateSource::ExactDictionary));
        assert!(d.decode("n'hao").iter().any(|c| c.text == "你好"
            && c.consumed == 5
            && c.source == CandidateSource::ExactDictionary));
        assert!(
            d.decode("nh'ao")
                .iter()
                .all(|c| c.text != "你好" || c.consumed == 3)
        );
        assert!(d.decode("zg").iter().any(|c| c.text == "中国"
            && c.consumed == 2
            && c.source == CandidateSource::ExactDictionary));
        assert!(
            d.decode("zg")
                .iter()
                .any(|c| c.text == "最高" && c.consumed == 2)
        );
        let zhg = d.decode("zhg");
        assert!(zhg.iter().any(|c| c.text == "中国"
            && c.consumed == 3
            && c.source == CandidateSource::ExactDictionary));
        assert!(zhg.iter().all(|c| c.text != "最高"));
        assert!(d.decode("zhongguo").iter().any(|c| c.text == "中国"
            && c.consumed == 8
            && c.source == CandidateSource::ExactDictionary));
        assert!(d.decode("sh").iter().all(|c| c.text != "上海"));
        assert!(d.decode("shh").iter().any(|c| c.text == "上海"
            && c.consumed == 3
            && c.source == CandidateSource::ExactDictionary));
        assert!(d.decode("nhshijie").iter().any(|c| c.text == "你好世界"
            && c.consumed == 8
            && c.source == CandidateSource::Composed));
        let mut cache = decoder::DecodeCache::default();
        for input in [
            "n", "nh", "nhx", "nhao", "n'h", "zg", "zhg", "nihao", "nih", "sh", "shh",
        ] {
            let cached = d.decode_cached(input, &mut cache).unwrap();
            let fresh = d.decode(input);
            assert_eq!(cached.len(), fresh.len(), "{input}");
            for (cached, fresh) in cached.iter().zip(&fresh) {
                assert_eq!(
                    (
                        cached.text.as_str(),
                        cached.consumed,
                        cached.source.clone(),
                        cached.score.to_bits()
                    ),
                    (
                        fresh.text.as_str(),
                        fresh.consumed,
                        fresh.source.clone(),
                        fresh.score.to_bits()
                    ),
                    "{input}"
                );
            }
        }
        let mut session = learning_session(Arc::new(d));
        session.insert("nhx").unwrap();
        let index = session
            .candidates()
            .iter()
            .position(|c| c.text == "你好" && c.consumed == 2)
            .unwrap();
        assert_eq!(session.select(index).as_deref(), Some("你好"));
        assert_eq!(session.input(), "x");
        session.reset();
        session.insert("nh").unwrap();
        assert_eq!(session.candidates()[0].text, "你好");
        assert_eq!(session.candidates()[0].source, CandidateSource::User);
    }
    #[test]
    fn english_dictionary_rows_do_not_pollute_pinyin() {
        let dictionary = "...\n公婆\tgong po\t100\n古朴\tgu pu\t1000\n加速\tjia su\t100\n";
        let before = Decoder::new(Dictionary::from_rime_reader(Cursor::new(dictionary)).unwrap());
        let after = Decoder::new(
            Dictionary::from_rime_reader(Cursor::new(format!("{dictionary}GPU\tgpu\t100\n")))
                .unwrap(),
        );
        for input in ["gp", "GPU", "gpujiasu"] {
            assert_eq!(before.decode(input), after.decode(input));
        }
        assert!(
            after
                .decode("gp")
                .iter()
                .any(|c| c.text == "公婆" && c.consumed == 2)
        );
    }
    #[test]
    fn prefix_selection_keeps_remainder_and_learns() {
        let mut s = learning_session(decoder());
        s.insert("nihao").unwrap();
        let index = s
            .candidates()
            .iter()
            .position(|c| c.text == "你" && c.consumed == 2)
            .unwrap();
        assert_eq!(s.select(index).as_deref(), Some("你"));
        assert_eq!(s.input(), "hao");
        assert_eq!(s.cursor(), 3);
        assert_eq!(s.context(), "你");
        s.reset();
        s.insert("ni").unwrap();
        assert_eq!(s.candidates()[0].source, CandidateSource::User);
    }
    #[test]
    fn shared_learning_and_completion_source_are_preserved() {
        let mut first = learning_session(decoder());
        first.insert("ni").unwrap();
        let index = first
            .candidates()
            .iter()
            .position(|c| c.text == "你")
            .unwrap();
        first.select(index).unwrap();
        let shared = Arc::new(Learning::new(false));
        let mut restored = Session::new(decoder(), Some(shared.clone()));
        restored.insert("ni").unwrap();
        restored.select(0).unwrap();
        let mut another = Session::new(decoder(), Some(shared));
        another.insert("ni").unwrap();
        assert_eq!(another.candidates()[0].source, CandidateSource::User);
        assert!(another.context().is_empty());
        first.reset();
        first.insert("nih").unwrap();
        let index = first
            .candidates()
            .iter()
            .position(|c| c.text == "你好")
            .unwrap();
        first.select(index).unwrap();
        first.insert("nih").unwrap();
        assert_eq!(
            first
                .candidates()
                .iter()
                .find(|c| c.text == "你好")
                .unwrap()
                .source,
            CandidateSource::Completion
        );
    }

    #[test]
    fn edits_and_focus_reset_reject_late_results() {
        let mut s = Session::new(decoder(), None);
        s.insert("nihao").unwrap();
        let old = s.snapshot();
        assert!(s.backspace());
        assert_eq!(s.input(), "niha");
        let current = s.candidates().to_vec();
        assert!(!s.apply_candidates(old.revision, old.candidates, Some(usize::MAX), &[]));
        assert_eq!(s.candidates(), current);
        s.move_cursor(-2);
        assert!(s.delete_forward());
        assert_eq!(s.input(), "nia");
        let old = s.snapshot();
        s.reset();
        assert!(s.input().is_empty());
        assert!(s.candidates().is_empty());
        assert!(s.context().is_empty());
        assert!(!s.apply_candidates(old.revision, old.candidates, Some(usize::MAX), &[]));
        assert!(s.candidates().is_empty());
    }
    #[test]
    fn partial_selection_keeps_remainder_and_refinement_preserves_ordinary_candidates() {
        let decoder = decoder();
        let mut session = Session::new(Arc::clone(&decoder), None);
        session.insert("nihaoshijie").unwrap();
        let hit = session
            .candidates()
            .iter()
            .find(|c| {
                c.text == "你好世界"
                    && c.consumed == session.input().len()
                    && c.source != CandidateSource::Completion
            })
            .unwrap();
        assert_eq!(hit.text, "你好世界");
        assert_eq!(hit.consumed, "nihaoshijie".len());
        assert_eq!(hit.source, CandidateSource::Composed);

        session.reset();
        session.insert("nihaoshij").unwrap();
        assert!(session.candidates().iter().any(|candidate| {
            candidate.text == "你好世界" && candidate.source == CandidateSource::Completion
        }));
        assert!(!session.candidates().iter().any(|c| c.text == "你好世界"
            && c.consumed == session.input().len()
            && c.source != CandidateSource::Completion));

        session.reset();
        session.insert("nihao").unwrap();
        let index = session
            .candidates()
            .iter()
            .position(|candidate| candidate.text == "你" && candidate.consumed == 2)
            .unwrap();
        assert_eq!(session.select(index).as_deref(), Some("你"));
        assert_eq!(session.input(), "hao");

        let mut session = Session::new(decoder.clone(), None);
        session.set_context("我去");
        session.insert("xian").unwrap();
        let candidates = session.candidates();
        let before: Vec<_> = candidates
            .iter()
            .filter(|candidate| {
                candidate.source != CandidateSource::Completion && !candidate.source.is_correction()
            })
            .map(|candidate| {
                (
                    candidate.text.clone(),
                    candidate.consumed,
                    candidate.source.clone(),
                    candidate.score.to_bits(),
                )
            })
            .collect();
        refine(&mut model(), &decoder, &mut session);
        let candidates = session.candidates();
        let after: Vec<_> = candidates
            .iter()
            .map(|candidate| {
                (
                    candidate.text.clone(),
                    candidate.consumed,
                    candidate.source.clone(),
                    candidate.score.to_bits(),
                )
            })
            .collect();
        for existing in before {
            assert!(after.contains(&existing), "refinement changed {existing:?}");
        }
    }

    #[test]
    fn invalid_input_is_atomic() {
        let mut s = Session::new(decoder(), None);
        s.insert("xian").unwrap();
        assert!(s.insert("中").is_err());
        assert_eq!(s.input(), "xian");
        assert!(s.insert(&"a".repeat(129)).is_err());
        assert_eq!(s.input(), "xian");
    }

    #[test]
    #[ignore = "requires downloaded full dictionary assets"]
    fn real_dictionary_loads_and_decodes() {
        let d = Decoder::new(Dictionary::load_dir("data/dictionaries").unwrap());
        assert!(d.dictionary().len() > 500_000);
        for input in [
            "nihao",
            "woaini",
            "zhonghuarenmingongheguo",
            "xian",
            "xi'an",
        ] {
            let candidates = d.decode(input);
            assert!(
                candidates.iter().any(|c| c.consumed == input.len()),
                "{input}"
            );
            println!(
                "{input}: {:?}",
                candidates.iter().take(5).collect::<Vec<_>>()
            );
        }
        assert!(
            d.decode("xian").iter().any(|c| c.text == "现"),
            "ambiguous one-syllable xian must survive beam"
        );
        assert!(d.decode("xian").iter().any(|c| c.text == "西安"));
        assert!(d.decode("xi'an").iter().any(|c| c.text == "西安"));
        assert!(!d.decode("xi'an").iter().any(|c| c.text == "现"));
        assert!(d.decode("n").iter().any(|c| c.text == "你"));
        assert!(d.decode("n").iter().any(|c| c.text == "年"));
        for input in ["l", "L"] {
            let candidates = d.decode(input);
            assert_eq!(candidates[0].text, "了", "{input}");
            assert_eq!(candidates[0].consumed, 1);
            assert_eq!(candidates[0].source, CandidateSource::ExactDictionary);
            assert!(candidates.iter().any(|c| c.text == "来"));
        }
        for (input, expected) in [("le", "了"), ("lai", "来"), ("dui", "对"), ("ru", "如")] {
            let candidates = d.decode(input);
            assert_eq!(candidates[0].text, expected, "{input}");
            assert_eq!(candidates[0].source, CandidateSource::ExactDictionary);
        }
        let nh = d.decode("nh");
        let hello = nh
            .iter()
            .find(|c| c.text == "你好")
            .expect("nh recalls 你好");
        assert_eq!(hello.consumed, 2);
        assert_eq!(hello.source, CandidateSource::ExactDictionary);
        let girl = nh.iter().position(|c| c.text == "女孩").expect("女孩");
        let hello_at = nh.iter().position(|c| c.text == "你好").unwrap();
        assert!(girl < hello_at);
        println!(
            "nh: 女孩@{girl} 你好@{hello_at} consumed={} {:?}",
            hello.consumed, hello.source
        );
        assert!(d.decode("nihao").iter().any(|c| c.text == "你好"
            && c.consumed == 5
            && c.source == CandidateSource::ExactDictionary));
        assert!(d.decode("nih").iter().any(|c| {
            c.text == "你好" && c.consumed == 3 && c.source == CandidateSource::Completion
        }));
        assert!(d.decode("n").iter().all(|c| c.text != "你好"));
        assert!(d.decode("nhao").iter().any(|c| c.text == "你好"
            && c.consumed == 4
            && c.source == CandidateSource::ExactDictionary));
        assert!(d.decode("zg").iter().any(|c| c.text == "中国"
            && c.consumed == 2
            && c.source == CandidateSource::ExactDictionary));
        assert!(d.decode("zhg").iter().any(|c| c.text == "中国"
            && c.consumed == 3
            && c.source == CandidateSource::ExactDictionary));
        assert!(d.decode("sh").iter().all(|c| c.text != "上海"));
        for (input, text) in [
            ("chongwuhua", "重物化"),
            ("chunxu", "纯虚"),
            ("dantai", "单态"),
            ("dantaihua", "单态化"),
            ("ditong", "低通"),
            ("duiyichu", "堆溢出"),
            ("jianxiangliang", "键向量"),
            ("jiaochashang", "交叉熵"),
            ("jingmoqi", "静默期"),
            ("qinrushi", "侵入式"),
            ("shangjiema", "熵解码"),
            ("tianchongfu", "填充符"),
            ("toumingdaye", "透明大页"),
            ("weidiaoyong", "尾调用"),
            ("yepianyi", "页偏移"),
            ("yichangzhi", "异常值"),
            ("yilaishu", "依赖树"),
            ("yinglianjieshu", "硬链接数"),
            ("yuanzixing", "原子性"),
            ("yulianjie", "预链接"),
            ("zhanhuisu", "栈回溯"),
            ("zhanjinsique", "栈金丝雀"),
            ("zhennei", "帧内"),
            ("zhenzhizhen", "帧指针"),
            ("zhuliuji", "驻留集"),
            ("zhuzhangliang", "逐张量"),
            ("zuzhen", "组帧"),
        ] {
            let found = d.decode(input).into_iter().find(|candidate| {
                candidate.text == text
                    && candidate.consumed == input.len()
                    && candidate.source == CandidateSource::ExactDictionary
            });
            assert!(found.is_some(), "{text} {input}");
            println!(
                "{input}\t{text}\tconsumed={}\t{:?}",
                input.len(),
                CandidateSource::ExactDictionary
            );
        }
        println!("entries={}", d.dictionary().len());
    }

    #[test]
    fn incremental_and_whole_input_sessions_produce_identical_refined_menus() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let decoder = Arc::new(Decoder::new(
            Dictionary::load_dir(root.join("data/dictionaries")).unwrap(),
        ));
        let (mut model, mut other) = (model(), model());
        let bits = |session: &Session| {
            session
                .candidates()
                .iter()
                .map(|candidate| {
                    (
                        candidate.text.clone(),
                        candidate.consumed,
                        candidate.source.clone(),
                        candidate.score.to_bits(),
                    )
                })
                .collect::<Vec<_>>()
        };
        let mut session = Session::new(Arc::clone(&decoder), None);
        let mut typed = Session::new(Arc::clone(&decoder), None);
        for text in ["n", "i", "h", "a", "o"] {
            session.insert(text).unwrap();
            typed.insert(session.input()).unwrap();
            assert_eq!(bits(&session), bits(&typed));
            assert!(!session.candidates().is_empty());
            typed.reset();
        }
        session.insert("'").unwrap();
        typed.insert(session.input()).unwrap();
        assert_eq!(bits(&session), bits(&typed));
        typed.reset();
        assert!(session.backspace());
        typed.insert(session.input()).unwrap();
        assert_eq!(session.input(), "nihao");
        assert_eq!(bits(&session), bits(&typed));
        session.set_context("昨天");
        typed.set_context("昨天");
        refine(&mut model, &decoder, &mut session);
        refine(&mut other, &decoder, &mut typed);
        assert_eq!(bits(&session), bits(&typed));
        let stale = session.snapshot();
        let partial = session
            .candidates()
            .iter()
            .position(|candidate| candidate.consumed < session.input().len())
            .expect("prefix candidate");
        let consumed = session.candidates()[partial].consumed;
        let selected = session.select(partial).expect("partial selection");
        assert_eq!(typed.select(partial).as_deref(), Some(selected.as_str()));
        assert_eq!(session.input(), &stale.input[consumed..]);
        assert!(!session.input().is_empty());
        assert_eq!(bits(&session), bits(&typed));
        assert!(!session.apply_candidates(stale.revision, stale.candidates, Some(usize::MAX), &[]));
        assert_eq!(bits(&session), bits(&typed));
    }

    #[test]
    fn dictionary_rejects_corrupt_rows_and_deduplicates() {
        assert!(Dictionary::from_rime_reader(Cursor::new("...\n坏行\n")).is_err());
        let d = Dictionary::from_rime_reader(Cursor::new("...\n你\tni\t10\n你\tni\t20\n")).unwrap();
        assert_eq!(d.len(), 1);
    }
    #[test]
    fn cpu_refinement_preserves_revision_prefix_and_learning() {
        let decoder = decoder();
        let mut session = learning_session(decoder.clone());
        session.set_context("我去");
        session.insert("xian").unwrap();
        let mut model = model();
        let before: Vec<_> = session
            .candidates()
            .iter()
            .filter(|candidate| {
                candidate.source != CandidateSource::Completion && !candidate.source.is_correction()
            })
            .cloned()
            .collect();
        let revision = session.revision();
        refine(&mut model, &decoder, &mut session);
        assert_eq!(session.revision(), revision);
        assert_eq!(session.input(), "xian");
        assert_eq!(session.context(), "我去");
        for existing in before {
            assert!(session.candidates().contains(&existing));
        }
        let prefix = session
            .candidates()
            .iter()
            .position(|c| c.text == "西" && c.consumed == 2)
            .unwrap();
        assert_eq!(session.select(prefix).as_deref(), Some("西"));
        assert_eq!(session.input(), "an");
        assert_eq!(session.context(), "我去西");
        session.reset();
        session.insert("xi").unwrap();
        assert_eq!(session.candidates()[0].source, CandidateSource::User);
        refine(&mut model, &decoder, &mut session);
        assert_eq!(session.candidates()[0].source, CandidateSource::User);
        let snap = session.snapshot();
        session.insert("a").unwrap();
        let current = session.candidates().to_vec();
        assert!(!session.apply_candidates(snap.revision, snap.candidates, Some(usize::MAX), &[]));
        assert_eq!(session.candidates(), current);
    }
}
