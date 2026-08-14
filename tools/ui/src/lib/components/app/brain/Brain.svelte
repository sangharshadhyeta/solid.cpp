<script lang="ts">
	/**
	 * Live per-expert moe-cache tier/heat map ("Brain" view), ported from
	 * Colibri's web/src/Brain.tsx (see docs/moe-cache-colibri-notes.md) to
	 * Svelte 5. Polls GET /experts every 1.5s and renders a Canvas 2D grid:
	 * one cell per (layer, expert), colour = cache tier, brightness = heat,
	 * and cells that were just written to flash white and decay back down -
	 * the same "living cortex" effect as the original.
	 *
	 * Simplified relative to the original: no expert-affinity atlas overlay
	 * (that's a separate, much bigger measured-topic-clustering system this
	 * fork doesn't have) and no i18n wiring for a first cut - plain English
	 * strings, matching how small a debugging/demo view this is meant to be.
	 */
	import { BrainCircuit, Flame } from '@lucide/svelte';
	import { ExpertsService } from '$lib/services';

	const TIER_LABELS = ['not cached', 'warm (probation)', 'hot (protected)'];
	const TIER_RGB: [number, number, number][] = [
		[58, 71, 80], // cold / not cached
		[90, 155, 216], // warm / probation
		[78, 214, 165] // hot / protected
	];
	const POLL_INTERVAL_MS = 1500;
	const PULSE_DECAY = 0.94;

	let canvasEl = $state<HTMLCanvasElement | null>(null);
	let wrapEl = $state<HTMLDivElement | null>(null);
	let wrapSize = $state({ w: 900, h: 520 });
	let rows = $state(0);
	let cols = $state(0);
	let mapBytes = $state<Uint8Array>(new Uint8Array());
	let probeErr = $state(false);
	let tip = $state<{ x: number; y: number; row: number; col: number; tier: number; heat: number } | null>(
		null
	);

	let pulse: Float32Array | null = null;
	let lastSeq = 0;
	let rafHandle = 0;
	let pollHandle: ReturnType<typeof setInterval> | undefined;
	let resizeObserver: ResizeObserver | undefined;

	function hexToBytes(hex: string): Uint8Array {
		const out = new Uint8Array(hex.length >> 1);

		for (let i = 0; i < out.length; i++) {
			out[i] = parseInt(hex.substr(i * 2, 2), 16) || 0;
		}

		return out;
	}

	async function poll() {
		try {
			const next = await ExpertsService.get();

			probeErr = false;

			if (!next.rows) {
				return;
			}

			rows = next.rows;
			cols = next.cols;
			mapBytes = hexToBytes(next.map);

			if (next.seq !== lastSeq && next.hits) {
				lastSeq = next.seq;

				const hitBytes = hexToBytes(next.hits);
				const n = rows * cols;

				if (!pulse || pulse.length !== n) {
					pulse = new Float32Array(n);
				}

				for (let i = 0; i < n; i++) {
					const byte = hitBytes[i >> 3] ?? 0;

					if (byte & (1 << (i & 7))) {
						pulse[i] = 1;
					}
				}
			}
		} catch {
			probeErr = true;
		}
	}

	function cellSize() {
		if (!cols || !rows) return { cell: 0, gap: 0 };

		const cell = Math.max(2, Math.floor(Math.min(wrapSize.w / cols, wrapSize.h / rows)));

		return { cell, gap: cell >= 4 ? 1 : 0 };
	}

	function draw() {
		const canvas = canvasEl;

		if (!canvas || !rows || !cols) return;

		const ctx = canvas.getContext('2d');

		if (!ctx) return;

		const { cell, gap } = cellSize();

		canvas.width = cols * (cell + gap);
		canvas.height = rows * (cell + gap);

		ctx.clearRect(0, 0, canvas.width, canvas.height);

		for (let r = 0; r < rows; r++) {
			for (let c = 0; c < cols; c++) {
				const i = r * cols + c;
				const byte = mapBytes[i] ?? 0;
				const tier = byte >> 6;
				const heat = byte & 63;
				const [R, G, B] = TIER_RGB[tier] ?? TIER_RGB[0];
				const lum = 0.35 + 0.65 * Math.min(heat / 24, 1);

				let rr = R * lum;
				let gg = G * lum;
				let bb = B * lum;

				const p = pulse ? pulse[i] : 0;

				if (p > 0.01) {
					rr += (255 - rr) * p;
					gg += (255 - gg) * p;
					bb += (255 - bb) * p;
				}

				ctx.fillStyle = `rgb(${rr | 0},${gg | 0},${bb | 0})`;
				ctx.fillRect(c * (cell + gap), r * (cell + gap), cell, cell);
			}
		}

		let alive = false;

		if (pulse) {
			for (let i = 0; i < pulse.length; i++) {
				if (pulse[i] > 0.01) {
					pulse[i] *= PULSE_DECAY;
					alive = true;
				} else {
					pulse[i] = 0;
				}
			}
		}

		if (alive) {
			rafHandle = requestAnimationFrame(draw);
		}
	}

	function onMove(e: MouseEvent) {
		if (!canvasEl || !rows || !cols) return;

		const rect = canvasEl.getBoundingClientRect();
		const scaleX = canvasEl.width / rect.width;
		const scaleY = canvasEl.height / rect.height;
		const { cell, gap } = cellSize();
		const col = Math.floor(((e.clientX - rect.left) * scaleX) / (cell + gap));
		const row = Math.floor(((e.clientY - rect.top) * scaleY) / (cell + gap));

		if (row < 0 || row >= rows || col < 0 || col >= cols) {
			tip = null;
			return;
		}

		const byte = mapBytes[row * cols + col] ?? 0;

		tip = { x: e.clientX, y: e.clientY, row, col, tier: byte >> 6, heat: byte & 63 };
	}

	const totals = $derived.by(() => {
		const t = [0, 0, 0];

		for (let i = 0; i < rows * cols; i++) {
			t[(mapBytes[i] ?? 0) >> 6]++;
		}

		return t;
	});

	$effect(() => {
		// Re-draw whenever a fresh poll updates the grid or the container resizes.
		void rows;
		void cols;
		void mapBytes;
		void wrapSize;

		cancelAnimationFrame(rafHandle);
		draw();
	});

	$effect(() => {
		void poll();
		pollHandle = setInterval(() => void poll(), POLL_INTERVAL_MS);

		return () => clearInterval(pollHandle);
	});

	$effect(() => {
		const el = wrapEl;

		if (!el) return;

		resizeObserver = new ResizeObserver(() => {
			wrapSize = { w: el.clientWidth - 24, h: el.clientHeight - 24 };
		});
		resizeObserver.observe(el);

		return () => resizeObserver?.disconnect();
	});
</script>

<div class="mx-auto flex h-full w-full max-w-5xl flex-col gap-4 p-4 md:p-8">
	<div class="flex flex-wrap items-center justify-between gap-3">
		<div class="flex items-center gap-2 text-lg font-semibold">
			<BrainCircuit class="size-5" />
			<span>Brain</span>
			<span class="text-muted-foreground text-sm font-normal">
				{#if rows && cols}
					— {rows} layers × {cols} experts
				{:else}
					— waiting for cache activity
				{/if}
			</span>
		</div>
		<div class="text-muted-foreground flex flex-wrap items-center gap-4 text-xs">
			<span class="flex items-center gap-1.5">
				<i class="inline-block size-2.5 rounded-full" style="background: #4ed6a5"></i>
				hot {totals[2].toLocaleString()}
			</span>
			<span class="flex items-center gap-1.5">
				<i class="inline-block size-2.5 rounded-full" style="background: #5a9bd8"></i>
				warm {totals[1].toLocaleString()}
			</span>
			<span class="flex items-center gap-1.5">
				<i class="inline-block size-2.5 rounded-full" style="background: #3a4750"></i>
				cold {totals[0].toLocaleString()}
			</span>
			<span class="flex items-center gap-1">
				<Flame class="size-3" />
				brightness = recent hits
			</span>
		</div>
	</div>

	<div
		bind:this={wrapEl}
		class="border-border bg-card relative min-h-[420px] flex-1 overflow-hidden rounded-lg border p-3"
	>
		{#if !rows || !cols}
			<p class="text-muted-foreground absolute inset-0 flex items-center justify-center text-sm">
				{#if probeErr}
					Could not reach /experts - is this build running the CUDA moe-cache backend?
				{:else}
					No expert activity cached yet. Send a request to the model to see the map populate.
				{/if}
			</p>
		{/if}
		<canvas
			bind:this={canvasEl}
			class="h-full w-full"
			onmousemove={onMove}
			onmouseleave={() => (tip = null)}
		></canvas>
	</div>

	{#if tip}
		<div
			class="border-border bg-popover text-popover-foreground fixed z-50 rounded-md border px-3 py-2 text-xs shadow-md"
			style:left="{Math.min(tip.x + 14, window.innerWidth - 220)}px"
			style:top="{Math.min(tip.y + 14, window.innerHeight - 100)}px"
		>
			<div class="font-medium">Layer {tip.row} · Expert {tip.col}</div>
			<div>Tier: <strong>{TIER_LABELS[tip.tier]}</strong></div>
			<div>Heat: <strong>{tip.heat === 0 ? 'never routed' : `${tip.heat} recent hits`}</strong></div>
		</div>
	{/if}
</div>
