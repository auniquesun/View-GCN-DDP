#!/usr/bin/env node --max-old-space-size=8192

var Evaluator = require('./lib/Evaluator.js');
var evaluator = new Evaluator();

var args = process.argv.slice(2);
if (args.length > 0) {
  var dir = args[0];
  evaluator.init(function() {
    evaluator.evaluate(dir);
  });
} else {
  console.log('Usage: provide directory with results (train/val/test x _normal/_perturbed)');
}