<script lang="ts">
	/**
	 * Live per-expert moe-cache tier/heat map ("Brain" view), ported from
	 * Colibri's web/src/Brain.tsx (see docs/moe-cache-colibri-notes.md) to
	 * Svelte 5. Polls GET /experts every 1.5s.
	 *
	 * When the server was started with --expert-atlas-file (output of
	 * tools/expert-atlas, offline-measured topic affinity per expert), the
	 * main view plots each expert by measured topic affinity - position from
	 * the atlas, colour from the live cache tier/heat - with the plain
	 * (layer, expert) grid tucked into a small picture-in-picture inset.
	 * Without an atlas file, the grid alone fills the main view, same as
	 * before.
	 */
	import { Flame } from '@lucide/svelte';
	import { ExpertsService } from '$lib/services';
	import { serverStore } from '$lib/stores/server.svelte';
	import type {
		ApiCoActivationEdge,
		ApiExpertAtlas,
		ApiExpertAtlasCell,
		ApiExpertMapStats
	} from '$lib/types/api';

	const TIER_LABELS = ['not cached', 'warm (probation)', 'hot (protected)'];
	const TIER_RGB: [number, number, number][] = [
		[58, 71, 80], // cold / not cached
		[90, 155, 216], // warm / probation
		[78, 214, 165] // hot / protected
	];
	const POLL_INTERVAL_MS = 1500;
	const PULSE_DECAY = 0.94;
	const ATLAS_POINT_R = 3.5;
	const PIP_W = 220;
	const PIP_H = 130;

	let canvasEl = $state<HTMLCanvasElement | null>(null);
	let pipCanvasEl = $state<HTMLCanvasElement | null>(null);
	let wrapEl = $state<HTMLDivElement | null>(null);
	let wrapSize = $state({ w: 900, h: 520 });
	let rows = $state(0);
	let cols = $state(0);
	let mapBytes = $state<Uint8Array>(new Uint8Array());
	let stats = $state<ApiExpertMapStats>({});
	let atlas = $state<ApiExpertAtlas | undefined>(undefined);
	let probeErr = $state(false);

	// Where the model actually lives, and why it was split that way. Static
	// per launch (unlike everything else here, which polls), so it reads from
	// the already-fetched /props rather than adding a second poll.
	let placement = $derived.by(() => {
		const solid = serverStore.props?.solid_cpp;
		const p = solid?.placement;
		if (!p || p.n_cpu_moe_final === undefined || !p.n_layer) {
			return null;
		}
		const onCpu = p.n_cpu_moe_final;
		const onGpu = Math.max(0, p.n_layer - onCpu);
		const requested = p.n_cpu_moe_requested ?? -1;
		return {
			onCpu,
			onGpu,
			nLayer: p.n_layer,
			nExpert: solid?.n_expert ?? 0,
			// Only a raise is worth explaining: it means the layout was chosen
			// for the context, not asked for.
			raisedFrom: requested >= 0 && onCpu > requested ? requested : null,
			nCtx: serverStore.props?.default_generation_settings?.n_ctx ?? p.n_ctx_requested ?? 0,
			nSlots: serverStore.props?.total_slots ?? 0
		};
	});
	let tip = $state<{
		x: number;
		y: number;
		row: number;
		col: number;
		tier: number;
		heat: number;
		topic?: string;
		spec?: number;
	} | null>(null);

	let pulse: Float32Array | null = null;
	let lastSeq = 0;
	let rafHandle = 0;
	let pollHandle: ReturnType<typeof setInterval> | undefined;
	let resizeObserver: ResizeObserver | undefined;
	let atlasPointsPx: { px: number; py: number; cell: ApiExpertAtlasCell }[] = [];

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
			stats = next.stats ?? {};
			atlas = next.atlas;

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

	function cellSize(w: number, h: number) {
		if (!cols || !rows) return { cell: 0, gap: 0 };

		const cell = Math.max(2, Math.floor(Math.min(w / cols, h / rows)));

		return { cell, gap: cell >= 4 ? 1 : 0 };
	}

	function cellColor(row: number, col: number): [number, number, number, number] {
		const i = row * cols + col;
		const byte = row < rows && col < cols ? (mapBytes[i] ?? 0) : 0;
		const tier = byte >> 6;
		const heat = byte & 63;
		const [R, G, B] = TIER_RGB[tier] ?? TIER_RGB[0];
		const lum = 0.35 + 0.65 * Math.min(heat / 24, 1);

		let rr = R * lum;
		let gg = G * lum;
		let bb = B * lum;

		const p = pulse && i < pulse.length ? pulse[i] : 0;

		if (p > 0.01) {
			rr += (255 - rr) * p;
			gg += (255 - gg) * p;
			bb += (255 - bb) * p;
		}

		return [rr | 0, gg | 0, bb | 0, tier];
	}

	function drawGrid(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, boxW: number, boxH: number) {
		const { cell, gap } = cellSize(boxW, boxH);

		canvas.width = cols * (cell + gap);
		canvas.height = rows * (cell + gap);
		// CSS must match the intrinsic pixel size exactly, otherwise h-full/w-full
		// stretches the canvas to the wrapper's box and squishes square cells into
		// tall rectangles.
		canvas.style.width = `${canvas.width}px`;
		canvas.style.height = `${canvas.height}px`;

		ctx.clearRect(0, 0, canvas.width, canvas.height);

		for (let r = 0; r < rows; r++) {
			for (let c = 0; c < cols; c++) {
				const [rr, gg, bb] = cellColor(r, c);

				ctx.fillStyle = `rgb(${rr},${gg},${bb})`;
				ctx.fillRect(c * (cell + gap), r * (cell + gap), cell, cell);
			}
		}
	}

	function drawAtlas(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement) {
		if (!atlas) return;

		canvas.width = wrapSize.w;
		canvas.height = wrapSize.h;
		canvas.style.width = `${canvas.width}px`;
		canvas.style.height = `${canvas.height}px`;

		ctx.clearRect(0, 0, canvas.width, canvas.height);

		const cx = canvas.width / 2;
		const cy = canvas.height / 2;
		const radius = Math.max(40, Math.min(canvas.width, canvas.height) / 2 - 48);

		// category labels around the rim, at the same anchor angles the
		// server used to compute each cell's (x, y)
		ctx.font = '11px sans-serif';
		ctx.fillStyle = 'rgba(148, 163, 184, 0.8)';
		ctx.textAlign = 'center';
		ctx.textBaseline = 'middle';
		// Discovered axes have no anchor geometry - a cell's x/y is not a
		// weighted average of category positions, so a ring of labels around
		// the disc would assert an adjacency that was never measured.
		const nCat = atlas.discovered ? 0 : atlas.categories.length;
		for (let c = 0; c < nCat; c++) {
			const angle = (2 * Math.PI * c) / nCat;
			const lx = cx + Math.cos(angle) * (radius + 22);
			const ly = cy + Math.sin(angle) * (radius + 22);
			ctx.fillText(atlas.categories[c], lx, ly);
		}

		ctx.strokeStyle = 'rgba(148, 163, 184, 0.15)';
		ctx.beginPath();
		ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
		ctx.stroke();

		atlasPointsPx = atlas.cells.map((cell) => {
			const px = cx + cell.x * radius;
			const py = cy + cell.y * radius;
			return { px, py, cell };
		});

		// Cold (never-cached) points are the vast majority of experts and carry
		// little signal - draw them as a faint, strokeless backdrop texture so
		// they don't visually bury the small number of experts actually hot in
		// the cache right now. Draw cold first, then warm, then hot on top, and
		// within a tier, more-specialized points last so a generalist blob
		// never fully covers a specialist's edge.
		const ordered = [...atlasPointsPx].sort((a, b) => {
			const [, , , tierA] = cellColor(a.cell.layer, a.cell.expert);
			const [, , , tierB] = cellColor(b.cell.layer, b.cell.expert);
			return tierA - tierB || a.cell.spec - b.cell.spec;
		});

		for (const { px, py, cell } of ordered) {
			const [rr, gg, bb, tier] = cellColor(cell.layer, cell.expert);

			if (tier === 0) {
				ctx.beginPath();
				ctx.arc(px, py, ATLAS_POINT_R * 0.6, 0, 2 * Math.PI);
				ctx.fillStyle = `rgba(${rr},${gg},${bb},0.28)`;
				ctx.fill();
				continue;
			}

			const r = ATLAS_POINT_R + 1 + cell.spec * 2.5;

			ctx.beginPath();
			ctx.arc(px, py, r, 0, 2 * Math.PI);
			ctx.fillStyle = `rgba(${rr},${gg},${bb},0.85)`;
			ctx.fill();
			ctx.lineWidth = 1;
			ctx.strokeStyle = 'rgba(226, 232, 240, 0.7)';
			ctx.stroke();
		}

		// Track 1 steps 4a/4b/5 (docs/plan.md): co-activation edges, drawn
		// faint underneath everything else already rendered above (points,
		// then the req_dir marker below, both stay legible on top). Endpoint
		// positions come from atlasPointsPx, keyed by (layer,expert) - the
		// edges themselves only carry layer/expert, not x/y, so this is a
		// lookup, not a direct plot. An edge whose endpoint isn't in this
		// atlas (shouldn't happen if server-side translation is correct, but
		// cheap to guard) is skipped rather than drawn from a wrong origin.
		const posOf = new Map<string, { px: number; py: number }>();
		for (const { px, py, cell } of atlasPointsPx) {
			posOf.set(`${cell.layer}:${cell.expert}`, { px, py });
		}
		const drawEdges = (
			edges: { layer_from: number; expert_from: number; layer_to: number; expert_to: number; count: number }[] | undefined,
			rgb: string
		) => {
			if (!edges?.length) return;
			const maxCount = Math.max(...edges.map((e) => e.count));
			for (const e of edges) {
				const a = posOf.get(`${e.layer_from}:${e.expert_from}`);
				const b = posOf.get(`${e.layer_to}:${e.expert_to}`);
				if (!a || !b) continue;
				const weight = maxCount > 0 ? e.count / maxCount : 0;
				ctx.beginPath();
				ctx.moveTo(a.px, a.py);
				ctx.lineTo(b.px, b.py);
				ctx.strokeStyle = `rgba(${rgb},${(0.06 + weight * 0.3).toFixed(3)})`;
				ctx.lineWidth = 0.5 + weight * 1.5;
				ctx.stroke();
			}
		};
		// Cross-layer first (a known gap - see api.d.ts - some of these are
		// same-layer artifacts, so they're drawn under, not over, the
		// within-layer edges) then within-layer on top, in a distinct hue.
		drawEdges(stats.co_activation_cross_layer, '90,155,216'); // warm blue
		drawEdges(stats.co_activation_within_layer, '78,214,165'); // hot green

		// Live request-direction marker (Track 1 step 0/1) - same (x,y)
		// space as every cell above, same projection. Clamped to the disc:
		// unlike a cell's x/y (bounded by construction, a weighted average
		// of unit vectors), req_dir is a raw decaying EMA and has no such
		// guarantee - drawing it outside the ring it's meant to sit in
		// would look like a bug even when the math is fine.
		if (stats.req_dir) {
			let { x: rx, y: ry } = stats.req_dir;
			const mag = Math.sqrt(rx * rx + ry * ry);
			if (mag > 1) {
				rx /= mag;
				ry /= mag;
			}
			const mx = cx + rx * radius;
			const my = cy + ry * radius;
			ctx.beginPath();
			ctx.arc(mx, my, 9, 0, 2 * Math.PI);
			ctx.strokeStyle = 'rgba(248, 113, 113, 0.9)';
			ctx.lineWidth = 2;
			ctx.stroke();
			ctx.beginPath();
			ctx.moveTo(mx - 5, my);
			ctx.lineTo(mx + 5, my);
			ctx.moveTo(mx, my - 5);
			ctx.lineTo(mx, my + 5);
			ctx.strokeStyle = 'rgba(248, 113, 113, 0.9)';
			ctx.lineWidth = 1.5;
			ctx.stroke();
		}
	}


	function draw() {
		const canvas = canvasEl;
		const useAtlas = !!atlas?.cells.length;

		if (!canvas || (!useAtlas && (!rows || !cols))) return;

		const ctx = canvas.getContext('2d');

		if (!ctx) return;

		if (useAtlas) {
			drawAtlas(ctx, canvas);

			if (rows && cols && pipCanvasEl) {
				const pipCtx = pipCanvasEl.getContext('2d');
				if (pipCtx) {
					drawGrid(pipCtx, pipCanvasEl, PIP_W, PIP_H);
				}
			}
		} else {
			drawGrid(ctx, canvas, wrapSize.w, wrapSize.h);
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

	function closestCategory(cell: ApiExpertAtlasCell): string | undefined {
		if (!atlas || atlas.discovered || !atlas.categories.length) return undefined;
		if (Math.hypot(cell.x, cell.y) < 0.05) return undefined; // too central to attribute

		const angle = Math.atan2(cell.y, cell.x);
		const nCat = atlas.categories.length;
		let best = 0;
		let bestDist = Infinity;

		for (let c = 0; c < nCat; c++) {
			const catAngle = (2 * Math.PI * c) / nCat;
			const d = Math.abs(Math.atan2(Math.sin(angle - catAngle), Math.cos(angle - catAngle)));
			if (d < bestDist) {
				bestDist = d;
				best = c;
			}
		}

		return atlas.categories[best];
	}

	function onMoveAtlas(e: MouseEvent) {
		if (!canvasEl) return;

		const rect = canvasEl.getBoundingClientRect();
		const scaleX = canvasEl.width / rect.width;
		const scaleY = canvasEl.height / rect.height;
		const mx = (e.clientX - rect.left) * scaleX;
		const my = (e.clientY - rect.top) * scaleY;

		let nearest: { px: number; py: number; cell: ApiExpertAtlasCell } | null = null;
		let bestDist = Infinity;

		for (const p of atlasPointsPx) {
			const d = Math.hypot(p.px - mx, p.py - my);
			if (d < bestDist) {
				bestDist = d;
				nearest = p;
			}
		}

		if (!nearest || bestDist > 10) {
			tip = null;
			return;
		}

		const { cell } = nearest;
		const byte = cell.layer < rows && cell.expert < cols ? (mapBytes[cell.layer * cols + cell.expert] ?? 0) : 0;

		tip = {
			x: e.clientX,
			y: e.clientY,
			row: cell.layer,
			col: cell.expert,
			tier: byte >> 6,
			heat: byte & 63,
			topic: closestCategory(cell),
			spec: cell.spec
		};
	}

	function onMove(e: MouseEvent) {
		if (atlas?.cells.length) {
			onMoveAtlas(e);
			return;
		}

		if (!canvasEl || !rows || !cols) return;

		const rect = canvasEl.getBoundingClientRect();
		const scaleX = canvasEl.width / rect.width;
		const scaleY = canvasEl.height / rect.height;
		const { cell, gap } = cellSize(wrapSize.w, wrapSize.h);
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
		void atlas;

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

<div class="mx-auto flex h-full w-full max-w-5xl flex-col gap-4 p-4 pt-16 md:p-8">
	<div class="flex flex-wrap items-center justify-between gap-3">
		<div class="flex items-center gap-2 text-lg font-semibold">
			<span>Brain</span>
			<span class="text-muted-foreground text-sm font-normal">
				{#if atlas?.cells.length}
					— {atlas.cells.length} experts positioned by
					{atlas.discovered ? 'discovered co-activation axes' : 'topic affinity'}
				{:else if rows && cols}
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
		class="border-border bg-card relative flex min-h-[420px] flex-1 items-center justify-center overflow-hidden rounded-lg border p-3"
	>
		{#if !atlas?.cells.length && (!rows || !cols)}
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
		{#if atlas?.cells.length && rows && cols}
			<div
				class="border-border bg-card/90 absolute right-3 bottom-3 overflow-hidden rounded-md border shadow-md backdrop-blur-sm"
			>
				<div class="text-muted-foreground border-border border-b px-2 py-1 text-[10px]">
					Grid — {rows}×{cols}
				</div>
				<canvas bind:this={pipCanvasEl} style:width="{PIP_W}px" style:height="{PIP_H}px"></canvas>
			</div>
		{/if}
	</div>

	{#if placement}
		<div class="border-border bg-card rounded-lg border p-3">
			<div class="mb-2 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
				<span class="text-xs font-medium">Placement</span>

				<span class="text-muted-foreground text-xs tabular-nums">
					{placement.nCtx.toLocaleString()} token context
					{#if placement.nSlots}· {placement.nSlots} slot{placement.nSlots === 1 ? '' : 's'}{/if}
				</span>
			</div>

			<!-- One bar for the whole model: GPU-resident layers vs layers whose
			     experts were pushed to CPU to make the context fit. -->
			<div class="bg-muted flex h-2.5 w-full overflow-hidden rounded-full">
				<div
					class="h-full"
					style:width="{(placement.onGpu / placement.nLayer) * 100}%"
					style:background-color="rgb(78, 214, 165)"
				></div>

				<div
					class="h-full"
					style:width="{(placement.onCpu / placement.nLayer) * 100}%"
					style:background-color="rgb(90, 155, 216)"
				></div>
			</div>

			<div class="text-muted-foreground mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs">
				<span class="inline-flex items-center gap-1.5">
					<span class="size-2 rounded-full" style:background-color="rgb(78, 214, 165)"></span>
					<strong class="text-foreground tabular-nums">{placement.onGpu}</strong> layer{placement.onGpu ===
					1
						? ''
						: 's'} on GPU
				</span>

				<span class="inline-flex items-center gap-1.5">
					<span class="size-2 rounded-full" style:background-color="rgb(90, 155, 216)"></span>
					<strong class="text-foreground tabular-nums">{placement.onCpu}</strong> with experts in CPU
					RAM
					{#if placement.nExpert}
						<span class="text-muted-foreground"
							>({(placement.onCpu * placement.nExpert).toLocaleString()} experts, cached on demand)</span
						>
					{/if}
				</span>
			</div>

			{#if placement.raisedFrom !== null}
				<div class="text-muted-foreground mt-2 text-xs">
					Raised from {placement.raisedFrom} to {placement.onCpu} CPU layer{placement.onCpu === 1
						? ''
						: 's'} so the requested {placement.nCtx.toLocaleString()}-token context would fit — the context
					was kept, the placement gave way.
				</div>
			{/if}
		</div>
	{/if}

	{#if stats.slots_total}
		<div class="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
			<div class="border-border bg-card rounded-lg border p-3">
				<div class="text-muted-foreground text-xs">Hit rate</div>
				<div class="text-lg font-semibold tabular-nums">
					{((stats.hit_rate ?? 0) * 100).toFixed(1)}%
				</div>
				<div class="text-muted-foreground text-xs tabular-nums">
					{(stats.hits ?? 0).toLocaleString()} / {((stats.hits ?? 0) + (stats.misses ?? 0)).toLocaleString()}
				</div>
			</div>
			<div class="border-border bg-card rounded-lg border p-3">
				<div class="text-muted-foreground text-xs">Slots used</div>
				<div class="text-lg font-semibold tabular-nums">
					{(stats.slots_used ?? 0).toLocaleString()} / {(stats.slots_total ?? 0).toLocaleString()}
				</div>
				<div class="text-muted-foreground text-xs tabular-nums">
					{(((stats.slots_used ?? 0) / Math.max(1, stats.slots_total ?? 1)) * 100).toFixed(0)}% full
				</div>
			</div>
			<div class="border-border bg-card rounded-lg border p-3">
				<div class="text-muted-foreground text-xs">Protected (hot)</div>
				<div class="text-lg font-semibold tabular-nums">{(stats.protected_slots ?? 0).toLocaleString()}</div>
				<div class="text-muted-foreground text-xs">of {(stats.slots_used ?? 0).toLocaleString()} resident</div>
			</div>
			<div class="border-border bg-card rounded-lg border p-3">
				<div class="text-muted-foreground text-xs">Avg heat</div>
				<div class="text-lg font-semibold tabular-nums">{(stats.avg_heat ?? 0).toFixed(1)}</div>
				<div class="text-muted-foreground text-xs">resident slots</div>
			</div>
			<div class="border-border bg-card rounded-lg border p-3">
				<div class="text-muted-foreground text-xs">VRAM used</div>
				<div class="text-lg font-semibold tabular-nums">
					{((stats.allocated_mib ?? 0) / 1024).toFixed(2)} GB
				</div>
				<div class="text-muted-foreground text-xs tabular-nums">
					of {((stats.budget_mib ?? 0) / 1024).toFixed(2)} GB budget
				</div>
			</div>
			<div class="border-border bg-card rounded-lg border p-3">
				<div class="text-muted-foreground text-xs">Evictions</div>
				<div class="text-lg font-semibold tabular-nums">{(stats.evictions ?? 0).toLocaleString()}</div>
				<div class="text-muted-foreground text-xs tabular-nums">
					{(stats.fill_failures ?? 0).toLocaleString()} fill failures
				</div>
			</div>
		</div>
	{/if}

	{#if tip}
		<div
			class="border-border bg-popover text-popover-foreground fixed z-50 rounded-md border px-3 py-2 text-xs shadow-md"
			style:left="{Math.min(tip.x + 14, window.innerWidth - 220)}px"
			style:top="{Math.min(tip.y + 14, window.innerHeight - 100)}px"
		>
			<div class="font-medium">Layer {tip.row} · Expert {tip.col}</div>
			<div>Tier: <strong>{TIER_LABELS[tip.tier]}</strong></div>
			<div>Heat: <strong>{tip.heat === 0 ? 'never routed' : `${tip.heat} recent hits`}</strong></div>
			{#if tip.topic}
				<div>Closest topic: <strong>{tip.topic}</strong></div>
			{/if}
			{#if tip.spec !== undefined}
				<div>Specialization: <strong>{(tip.spec * 100).toFixed(0)}%</strong></div>
			{/if}
		</div>
	{/if}
</div>
