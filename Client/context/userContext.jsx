/* eslint-disable react/prop-types */
import axios from "axios";
import { createContext, useState, useEffect } from "react";
import toast from "react-hot-toast";

export const UserContext = createContext({});

export function UserContextProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchUser = async () => {
      const token = localStorage.getItem("token");
      const storedUser = JSON.parse(localStorage.getItem("user")); // Load user data from localStorage

      if (token) {
        axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;

        if (storedUser) {
          setUser(storedUser); // Set the user data from localStorage
        } else {
          try {
            const { data } = await axios.get("/api/profile");
            setUser(data);
            localStorage.setItem("user", JSON.stringify(data)); // Save user data to localStorage
          } catch (error) {
            console.error("Error fetching user data:", error);
          }
        }
      }
      setLoading(false);
    };

    fetchUser();
  }, []);

  const clearCookie = async () => {
    try {
      await axios.post("/api/logout");
      localStorage.removeItem("token");
      localStorage.removeItem("user"); // Remove user data from localStorage
      toast.success("Logout Successfully");
      setUser(null);
    } catch (error) {
      console.error("Error clearing cookie:", error);
    }
  };

  return (
    <UserContext.Provider value={{ user, setUser, clearCookie, loading }}>
      {children}
    </UserContext.Provider>
  );
}
