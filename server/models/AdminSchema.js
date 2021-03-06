const mongoose = require("mongoose");

const adminSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
  },
  email: {
    type: String,
    required: true,
  },
  phone: {
    type: Number,
    required: true,
  },
  password: {
    type: String,
    required: true,
  },
  admins: [
    {
      admin:{
        type: mongoose.Schema.Types.ObjectId,
      ref: "ADMIN",
      },
      password:String
    },
  ],
});

const Admin = mongoose.model("ADMIN", adminSchema);

module.exports = Admin;
