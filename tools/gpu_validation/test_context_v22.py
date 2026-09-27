"""Source wiring regressions only; not execution of Rust or driver functions."""
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[2]
class ContextWiringTests(unittest.TestCase):
    def setUp(self):
        self.server=(ROOT/'ruda-driver-cuda/src/execution/server.rs').read_text()
        self.graph=(ROOT/'ruda-driver-cuda/src/execution/server/graph.rs').read_text()
    def test_command_propagates_context_error(self):
        body=self.server.split('fn command<',1)[1].split('fn flush_errors',1)[0]
        self.assertIn('self.set_current_checked()?;',body)
        self.assertLess(body.index('self.set_current_checked()?;'),body.index('self.streams.resolve'))
    def test_context_helper_does_not_unwrap(self):
        body=self.server.split('fn set_current_checked',1)[1].split('fn command<',1)[0]
        self.assertIn('map_err',body);self.assertNotIn('.unwrap()',body)
    def test_graph_no_duplicate_raw_context_set(self):
        self.assertNotIn('self.ctx.unsafe_set_current()',self.graph)
    def test_close_still_establishes_context(self):
        body=self.graph.split('GraphCommand::Close => {',1)[1].split('},',1)[0]
        self.assertIn('self.set_current_checked()?;',body)
        self.assertIn('entry.native.close()',body)
    def test_dependency_and_pointer_checks_retained(self):
        self.assertIn('pins.iter().map(|p| &p.binding)',self.graph)
        self.assertIn('resource.ptr != pin.pointer',self.graph)
        self.assertIn('if prepared.is_empty() && !replay { return Ok(()); }',self.graph)
if __name__=='__main__': unittest.main()
