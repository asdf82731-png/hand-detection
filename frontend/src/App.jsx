import {useEffect, useRef, useState} from 'react';
import {Badge} from '@astryxdesign/core/Badge';
import {Banner} from '@astryxdesign/core/Banner';
import {Button} from '@astryxdesign/core/Button';
import {Card} from '@astryxdesign/core/Card';
import {Collapsible} from '@astryxdesign/core/Collapsible';
import {FileInput} from '@astryxdesign/core/FileInput';
import {Grid} from '@astryxdesign/core/Grid';
import {Heading} from '@astryxdesign/core/Heading';
import {List, ListItem} from '@astryxdesign/core/List';
import {NumberInput} from '@astryxdesign/core/NumberInput';
import {ProgressBar} from '@astryxdesign/core/ProgressBar';
import {Section} from '@astryxdesign/core/Section';
import {Stack} from '@astryxdesign/core/Stack';
import {Table} from '@astryxdesign/core/Table';
import {Text} from '@astryxdesign/core/Text';
import {IS_DEMO, analyzeVideo, expectedCountsJson, translateSuggestion} from './api.js';
import {HANDS, STROKES} from './strokes.js';
import {loadHistory, saveHistory, clearHistory} from './history.js';

// A hand detected in fewer frames than this gives counts that are mostly guessed.
const LOW_DETECTION = 0.6;
const EMPTY_TARGETS = {left: {}, right: {}};

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [targets, setTargets] = useState(EMPTY_TARGETS);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState(loadHistory);
  const [installPrompt, setInstallPrompt] = useState(null);
  const cameraRef = useRef(null);
  const resultsRef = useRef(null);
  const busy = progress !== null && progress.fraction < 1;

  useEffect(() => {
    const onPrompt = (event) => { event.preventDefault(); setInstallPrompt(event); };
    window.addEventListener('beforeinstallprompt', onPrompt);
    return () => window.removeEventListener('beforeinstallprompt', onPrompt);
  }, []);
  useEffect(() => () => previewUrl && URL.revokeObjectURL(previewUrl), [previewUrl]);
  useEffect(() => { if (result) resultsRef.current?.scrollIntoView({behavior: 'smooth', block: 'start'}); }, [result]);

  function choose(chosen) {
    const picked = Array.isArray(chosen) ? chosen[0] : chosen;
    setFile(picked || null);
    setPreviewUrl(picked ? URL.createObjectURL(picked) : null);
    setError(null);
  }

  async function analyze() {
    setError(null);
    let expected;
    try { expected = expectedCountsJson(targets); } catch (err) { setError(err.message); return; }
    try {
      const data = await analyzeVideo(file, expected, setProgress);
      setResult(data);
      setHistory(saveHistory(file.name, data));
    } catch (err) {
      setProgress(null);
      setError(err.message);
    }
  }

  return (
    <Stack align="center" paddingInline={4} paddingBlock={4}>
    <Stack gap={4} width="100%" maxWidth={560}>
      <Stack direction="horizontal" gap={2} vAlign="center" hAlign="between">
        <Heading level={1}>鼓棒揮擊計數</Heading>
        {installPrompt && (
          <Button label="安裝到主畫面" variant="ghost" size="sm"
            onClick={async () => { installPrompt.prompt(); await installPrompt.userChoice; setInstallPrompt(null); }} />
        )}
      </Stack>

      {IS_DEMO && (
        <Banner status="info" collapsible={false} title="介面示範"
          description="這個版本沒有連到分析伺服器：影片不會上傳，結果是固定的示意資料。準確率會照你填的練習目標計算。" />
      )}

      <Section variant="muted">
        <Stack gap={3}>
          <Heading level={2}>1. 錄下打鼓的手</Heading>
          <Text type="supporting">手機橫放在正前方，雙手、鼓棒與雙肘都要入鏡。</Text>
          <Button label="錄影" variant="primary" size="lg" width="100%" onClick={() => cameraRef.current?.click()} />
          {/* Astryx's FileInput cannot open the camera directly; only a native input with capture does on phones. */}
          <input ref={cameraRef} type="file" accept="video/*" capture="environment" hidden
            onChange={(e) => { choose(e.target.files?.[0]); e.target.value = ''; }} />
          <FileInput label="或從相簿選擇影片" accept="video/*" value={file} onChange={choose}
            placeholder="選擇影片" description={file ? `${(file.size / 1048576).toFixed(1)} MB` : undefined} />
          {/* No Astryx video component: a native player is needed to preview the clip. */}
          {previewUrl && <video src={previewUrl} controls playsInline style={{width: '100%', maxHeight: '50vh', borderRadius: 'var(--radius-container)'}} />}
          <Collapsible trigger={<Text weight="semibold">拍攝要點</Text>} defaultIsOpen={false}>
            <List listStyle="disc" density="compact">
              <ListItem label={<Text>鏡頭在演奏者正前方，約胸口高度、距離 1.5 公尺，水平拍攝</Text>} />
              <ListItem label={<Text>橫拍；雙手、鼓棒、雙肘都完整入鏡</Text>} />
              <ListItem label={<Text>光線充足、手上沒有強烈陰影；背景單純</Text>} />
              <ListItem label={<Text>手機固定不動（腳架或靠穩）</Text>} />
              <ListItem label={<Text>開始與結束時雙手各靜止 2 秒</Text>} />
            </List>
          </Collapsible>
        </Stack>
      </Section>

      <Section variant="muted">
        <Stack gap={3}>
          <Heading level={2}>2. 練習目標（選填）</Heading>
          <Text type="supporting">填入每種揮擊打了幾下，就能算出準確率。不填也可以，只看次數。</Text>
          {HANDS.map(([hand, handName]) => (
            <Stack key={hand} gap={1}>
              <Text type="label">{handName}</Text>
              <Grid columns={4} gap={2}>
                {STROKES.map(([stroke, name]) => (
                  <NumberInput key={stroke} label={name} size="sm" min={0} step={1} placeholder="–"
                    value={targets[hand][stroke] ?? null}
                    onChange={(value) => setTargets((t) => ({...t, [hand]: {...t[hand], [stroke]: value}}))} />
                ))}
              </Grid>
            </Stack>
          ))}
        </Stack>
      </Section>

      <Button label="3. 開始分析" variant="primary" size="lg" width="100%" isDisabled={!file} isLoading={busy} onClick={analyze} />
      {progress && (
        <ProgressBar label={progress.text} value={Math.round(progress.fraction * 100)} hasValueLabel />
      )}
      {error && <Banner status="error" title="無法分析" description={error} collapsible={false} />}

      {result && <Results data={result} innerRef={resultsRef} />}

      <Stack gap={2}>
        <Stack direction="horizontal" hAlign="between" vAlign="center">
          <Heading level={2}>練習紀錄</Heading>
          {history.length > 0 && <Button label="清除" variant="ghost" size="sm" onClick={() => setHistory(clearHistory())} />}
        </Stack>
        {history.length === 0 ? (
          <Text type="supporting">分析過的練習會列在這裡，只存在這支手機上。</Text>
        ) : (
          <List hasDividers density="compact">
            {history.map((h) => (
              <ListItem key={h.at} label={new Date(h.at).toLocaleString('zh-TW', {month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit'})}
                description={<Text type="supporting">左 {summary(h.left)}<br />右 {summary(h.right)}</Text>} endContent={h.name ? <Text type="supporting" maxLines={1}>{h.name}</Text> : null} />
            ))}
          </List>
        )}
      </Stack>
    </Stack>
    </Stack>
  );
}

function summary(counts) {
  return STROKES.map(([key, name]) => `${name}${counts?.[key] || 0}`).join(' ');
}

function Results({data, innerRef}) {
  const suggestions = (data.suggestions || []).map(translateSuggestion);
  return (
    <Stack gap={4} ref={innerRef}>
      <Heading level={2}>結果</Heading>
      {HANDS.map(([hand, handName]) => {
        const rate = data.detection_rate?.[hand];
        return rate !== undefined && rate < LOW_DETECTION ? (
          <Banner key={hand} status="warning" collapsible={false} title={`${handName}只在 ${Math.round(rate * 100)}% 的畫面中被偵測到`}
            description="次數可能不準。請照「拍攝要點」重拍。" />
        ) : null;
      })}
      {HANDS.map(([hand, handName]) => <HandCounts key={hand} data={data} hand={hand} handName={handName} />)}
      <Collapsible trigger={<Text weight="semibold">動作細節</Text>} defaultIsOpen={false}>
        <Stack gap={3}>
          <Text type="supporting">高度與下擊速度以手掌長度為單位；括號內是同一種揮擊之間的差異，越小越穩定。下擊速度代替力量，相機量不到力量。</Text>
          {HANDS.map(([hand, handName]) => <Metrics key={hand} metrics={data.stroke_metrics?.[hand]} handName={handName} />)}
        </Stack>
      </Collapsible>
      {suggestions.length > 0 && (
        <Stack gap={1}>
          <Heading level={3}>建議</Heading>
          <List listStyle="disc" density="compact">
            {/* A ReactNode label wraps; a plain string would be cut to one line. */}
            {suggestions.map((s) => <ListItem key={s} label={<Text>{s}</Text>} />)}
          </List>
        </Stack>
      )}
      <Text type="supporting">次數是模型的估計；目前的模型只用少量影片訓練，請把結果當參考。</Text>
    </Stack>
  );
}

function HandCounts({data, hand, handName}) {
  const counts = data[`${hand}_hand_counts`] || {};
  const accuracy = data[`${hand}_hand_accuracy`] || {};
  // Accuracy is shown only against the user's own targets: without them there
  // is no actual count, and the self-consistency score measures nothing real.
  const fromTargets = data.accuracy_basis?.[hand] === 'expected_counts';
  return (
    <Stack gap={2}>
      <Heading level={3}>{handName}</Heading>
      <Grid columns={4} gap={2}>
        {STROKES.map(([stroke, name, variant]) => (
          <Card key={stroke} variant={variant} padding={2}>
            <Stack gap={0.5} align="center">
              <Text type="display-3" hasTabularNumbers>{String(counts[stroke] ?? 0)}</Text>
              <Text type="supporting">{name}</Text>
              {fromTargets && accuracy[stroke] != null && <Badge variant="neutral" label={`${Math.round(accuracy[stroke])}%`} />}
            </Stack>
          </Card>
        ))}
      </Grid>
      {!fromTargets && <Text type="supporting">填入練習目標才能計算準確率。</Text>}
    </Stack>
  );
}

function fmt(mean, cv) {
  if (mean == null) return '–';
  return cv == null ? String(mean) : `${mean}（${cv}）`;
}

function Metrics({metrics, handName}) {
  if (!metrics) return null;
  const rows = STROKES.filter(([key]) => metrics.strokes?.[key]?.count).map(([key, name]) => {
    const s = metrics.strokes[key];
    return {id: key, stroke: name, count: s.count, height: fmt(s.peak_height_mean, s.peak_height_cv),
      angle: fmt(s.angle_range_deg_mean, s.angle_range_deg_cv), speed: fmt(s.peak_down_speed_mean, s.peak_down_speed_cv)};
  });
  return (
    <Stack gap={1}>
      <Text type="label">{handName}</Text>
      {rows.length === 0 ? (
        <Text type="supporting">沒有偵測到揮擊</Text>
      ) : (
        <>
          <Table idKey="id" density="compact" data={rows} columns={[
            {key: 'stroke', header: '揮擊'},
            {key: 'count', header: '次數', align: 'end'},
            {key: 'height', header: '最高高度', align: 'end'},
            {key: 'angle', header: '夾角幅度°', align: 'end'},
            {key: 'speed', header: '下擊速度', align: 'end'},
          ]} />
          <Text type="supporting">揮擊間隔：{fmt(metrics.interval_s_mean, metrics.interval_s_cv)} 秒</Text>
        </>
      )}
    </Stack>
  );
}
