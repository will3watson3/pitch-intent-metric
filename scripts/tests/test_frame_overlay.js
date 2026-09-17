// Run with Node.js or JavaScriptCore jsc from the repository root.
const source = typeof readFile === 'function'
  ? readFile('frame-review/app.js')
  : require('node:fs').readFileSync('frame-review/app.js', 'utf8');
new Function(source); // Compile the complete UI, including event handlers.
function implementation(name) {
  const start = source.indexOf(`function ${name}(`);
  const end = source.indexOf('\nfunction ', start + 1);
  if (start < 0) throw new Error(`Missing implementation: ${name}`);
  return source.slice(start, end < 0 ? undefined : end);
}
const names = ['num', 'text', 'clamp', 'normalizeOverlay', 'gameKey', 'automatedOverlayForFrame', 'savedOverlayForFrame'];
const factory = new Function('state', 'defaultOverlay', names.map(implementation).join('\n') + '\nreturn { automatedOverlayForFrame, savedOverlayForFrame };');
const state = { calibrations: { games: {}, default: { left:43,top:22,width:14,height:31 } } };
const api = factory(state, state.calibrations.default);
function assert(condition, message) { if (!condition) throw new Error(message); }
const frame = { frame_uid:'p_2',game_pk:'game',zone_x:640,zone_y:300,zone_width:43,zone_height:55 };
const automatic = api.automatedOverlayForFrame(frame);
assert(Math.abs(automatic.width-43/1280*100)<1e-9, 'Small zones must not be inflated');
state.calibrations.games.game = { reference_frame_uid:'p_1',left:10,top:10,width:20,height:30,frame_override:true };
assert(api.savedOverlayForFrame(frame).width === automatic.width, 'Other-frame calibration must not override this frame');
state.calibrations.games.game.reference_frame_uid = 'p_2';
assert(api.savedOverlayForFrame(frame).width === 20, 'Explicit manual correction must stay scoped to its frame');
state.calibrations.games.game.frame_override = false;
assert(api.savedOverlayForFrame(frame).width === automatic.width, 'Old game reference must not override refined geometry');
const scaled = api.automatedOverlayForFrame({ ...frame, image_width:2560,image_height:1440,zone_x:1280,zone_y:600,zone_width:86,zone_height:110 });
assert(scaled.width === automatic.width && scaled.top === automatic.top, 'Resolution must preserve overlay placement');
assert(api.automatedOverlayForFrame({ ...frame,zone_width:0 }) === null, 'Zero-width zones must be hidden');
if (typeof print === 'function') print('PASS: full UI syntax and 6 frame-overlay regression checks');
else console.log('PASS: full UI syntax and 6 frame-overlay regression checks');
